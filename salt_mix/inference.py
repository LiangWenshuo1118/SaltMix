from __future__ import annotations

import csv
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .chemistry import classify_component, tokenize_components
from .config_utils import resolve_model_rk_order, resolve_model_temperature_law, resolve_model_use_phy
from .dataset import PropertyDataset
from .preprocess import build_axis_points, load_config, load_json_with_fallback
from .properties import get_property_spec, parse_composition, parse_temperature_point, parse_temperature_range
from .train import _lazy_torch

COVERAGE_DETAIL_LIMIT = 25
OUTPUT_DECIMAL_PLACES = 3


def _load_model(config: Dict[str, object], artifact_dir: Path, checkpoint_path: Optional[Path]):
    torch, _, _ = _lazy_torch()
    from .model import PropertyRKModel

    with (artifact_dir / "vocab.json").open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)
    if checkpoint_path is None:
        checkpoint_path = artifact_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    temperature_law = resolve_model_temperature_law(config["model"])
    model = PropertyRKModel(
        vocab_size=len(vocab),
        embed_dim=int(config["model"]["embed_dim"]),
        encoder_dim=int(config["model"]["encoder_dim"]),
        num_heads=int(config["model"]["num_heads"]),
        num_layers=int(config["model"]["num_layers"]),
        ff_dim=int(config["model"]["ff_dim"]),
        subnet_dim=int(config["model"]["subnet_dim"]),
        rk_order=resolve_model_rk_order(config["model"]),
        use_phy=resolve_model_use_phy(config["model"]),
        temperature_law=temperature_law,
        max_components=PropertyDataset(artifact_dir, "test").max_components,
        dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def _prediction_temperatures(raw_temperature: str, temperatures_per_record: int) -> List[float]:
    try:
        return [parse_temperature_point(raw_temperature)]
    except ValueError:
        lo, hi = parse_temperature_range(raw_temperature)
        return build_axis_points(lo, hi, temperatures_per_record)


def _prediction_rows(
    rows: List[Dict[str, object]],
    property_name: str,
    temperatures_per_record: int,
) -> List[Dict[str, object]]:
    samples: List[Dict[str, object]] = []
    for row in rows:
        data = row["data"]
        record_id = str(row.get("index", ""))
        system = str(data["system"]).strip()
        components = [component.strip() for component in system.split("-")]
        for component in components:
            reason = classify_component(component)
            if reason:
                raise ValueError(f"{record_id}: unsupported component {component}: {reason}")
        mole_fractions = parse_composition(str(data["composition"]), len(components))
        tokenized_components = tokenize_components(components)
        for temperature in _prediction_temperatures(str(data["temperature"]), temperatures_per_record):
            sample = {
                "record_id": record_id,
                "system": system,
                "components": components,
                "component_tokens": [item.tokens for item in tokenized_components],
                "component_parser_warnings": [item.parser_warning for item in tokenized_components],
                "mole_fractions": mole_fractions,
                "temperature": temperature,
                "target_value": 1.0,
            }
            if get_property_spec(property_name).supports_log_metrics:
                sample["target_log_value"] = 0.0
            samples.append(sample)
    return samples


def _format_mole_fractions(mole_fractions: Sequence[float]) -> str:
    return "-".join(f"{value:.10g}" for value in mole_fractions)


def _format_mole_percent(mole_fractions: Sequence[float]) -> str:
    return "-".join(f"{100.0 * value:.{OUTPUT_DECIMAL_PLACES}f}" for value in mole_fractions)


def _format_output_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.{OUTPUT_DECIMAL_PLACES}f}"


def _format_value_with_unit(value: Optional[float], unit: str) -> str:
    formatted_value = _format_output_float(value)
    if not formatted_value:
        return ""
    return f"{formatted_value} {unit}"


def _round_public_float(value: float) -> float:
    return round(float(value), OUTPUT_DECIMAL_PLACES)


def _proper_subsystems(components: Sequence[str]) -> List[tuple[str, ...]]:
    subsystems: List[tuple[str, ...]] = []
    component_indices = range(len(components))
    for size in range(1, len(components)):
        for indices in combinations(component_indices, size):
            subsystems.append(tuple(components[index] for index in indices))
    return subsystems


def _build_subsystem_index(samples: Sequence[Dict[str, object]]) -> Dict[frozenset[str], List[Dict[str, object]]]:
    index: Dict[frozenset[str], List[Dict[str, object]]] = {}
    for sample in samples:
        components = [str(component) for component in sample.get("components", [])]
        if not components:
            continue
        index.setdefault(frozenset(components), []).append(sample)
    return index


def _load_coverage_samples(artifact_dir: Path) -> List[Dict[str, object]]:
    train_samples_path = artifact_dir / "train_samples.json"
    if not train_samples_path.exists():
        return []
    with train_samples_path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        return []
    return samples


def _limited_values(values: Sequence[object], limit: int = COVERAGE_DETAIL_LIMIT) -> Dict[str, object]:
    return {
        "count": len(values),
        "values": list(values[:limit]),
        "truncated": len(values) > limit,
    }


def _composition_point(sample: Dict[str, object], subsystem: Sequence[str]) -> Dict[str, float]:
    components = [str(component) for component in sample.get("components", [])]
    mole_fractions = [float(value) for value in sample.get("mole_fractions", [])]
    composition_by_component = {
        component: 100.0 * mole_fraction
        for component, mole_fraction in zip(components, mole_fractions)
    }
    return {
        component: _round_public_float(composition_by_component[component])
        for component in subsystem
        if component in composition_by_component
    }


def _project_subsystem_composition(
    subsystem: Sequence[str],
    composition_by_component: Dict[str, float],
) -> Dict[str, float]:
    subtotal = sum(composition_by_component.get(component, 0.0) for component in subsystem)
    if subtotal <= 0.0:
        return {}
    return {
        component: _round_public_float(100.0 * composition_by_component.get(component, 0.0) / subtotal)
        for component in subsystem
    }


def _summarize_subsystem(
    requested_subsystem: Sequence[str],
    matching_samples: Sequence[Dict[str, object]],
    query_composition_mol_pct: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    subsystem_name = "-".join(requested_subsystem)
    query_subsystem_composition = _project_subsystem_composition(
        requested_subsystem,
        query_composition_mol_pct or {},
    )
    if not matching_samples:
        return {
            "subsystem": subsystem_name,
            "components": list(requested_subsystem),
            "query_subsystem_composition_mol_pct": query_subsystem_composition,
            "covered": False,
            "sample_count": 0,
            "record_count": 0,
            "systems": [],
        }

    record_ids = sorted({str(sample.get("record_id", "")) for sample in matching_samples})
    systems = sorted({str(sample.get("system", "")) for sample in matching_samples})
    temperatures = sorted({_round_public_float(float(sample["temperature"])) for sample in matching_samples})

    composition_points_by_key: Dict[tuple[tuple[str, float], ...], Dict[str, float]] = {}
    component_values: Dict[str, List[float]] = {component: [] for component in requested_subsystem}
    for sample in matching_samples:
        point = _composition_point(sample, requested_subsystem)
        for component, value in point.items():
            component_values[component].append(value)
        key = tuple((component, point.get(component, math.nan)) for component in requested_subsystem)
        composition_points_by_key[key] = point

    composition_points = [
        composition_points_by_key[key]
        for key in sorted(composition_points_by_key)
    ]
    composition_ranges = {
        component: {
            "min_mol_pct": _round_public_float(min(values)),
            "max_mol_pct": _round_public_float(max(values)),
        }
        for component, values in component_values.items()
        if values
    }
    return {
        "subsystem": subsystem_name,
        "components": list(requested_subsystem),
        "query_subsystem_composition_mol_pct": query_subsystem_composition,
        "covered": True,
        "sample_count": len(matching_samples),
        "record_count": len(record_ids),
        "systems": systems,
        "record_ids": _limited_values(record_ids),
        "temperature_range_K": {
            "min": min(temperatures),
            "max": max(temperatures),
        },
        "temperature_points_K": _limited_values(temperatures),
        "composition_ranges_mol_pct": composition_ranges,
        "composition_points_mol_pct": _limited_values(composition_points),
    }


def _coverage_summary_text(covered_count: int, total_count: int, missing_subsystems: Sequence[str]) -> str:
    if total_count == 0:
        return "no lower-order subsystems required"
    if not missing_subsystems:
        return f"{covered_count}/{total_count} lower-order subsystems covered"
    displayed_missing = list(missing_subsystems[:8])
    suffix = ""
    if len(missing_subsystems) > len(displayed_missing):
        suffix = f"; +{len(missing_subsystems) - len(displayed_missing)} more"
    return (
        f"{covered_count}/{total_count} lower-order subsystems covered; "
        f"missing: {', '.join(displayed_missing)}{suffix}"
    )


def _build_subsystem_coverage_report(
    sample: Dict[str, object],
    subsystem_index: Dict[frozenset[str], List[Dict[str, object]]],
    coverage_source: str = "train_samples.json",
) -> Dict[str, object]:
    components = [str(component) for component in sample["components"]]
    composition_mol_pct = {
        component: _round_public_float(100.0 * float(fraction))
        for component, fraction in zip(components, sample["mole_fractions"])
    }
    subsystems = _proper_subsystems(components)
    subsystem_reports = [
        _summarize_subsystem(
            subsystem,
            subsystem_index.get(frozenset(subsystem), []),
            composition_mol_pct,
        )
        for subsystem in subsystems
    ]
    missing_subsystems = [
        str(report["subsystem"])
        for report in subsystem_reports
        if not bool(report["covered"])
    ]
    covered_count = len(subsystem_reports) - len(missing_subsystems)
    total_count = len(subsystem_reports)
    coverage_fraction = _round_public_float(covered_count / total_count if total_count else 1.0)
    return {
        "record_id": sample["record_id"],
        "system": sample["system"],
        "components": components,
        "mole_fractions": list(sample["mole_fractions"]),
        "composition_mol_pct": composition_mol_pct,
        "coverage_source": coverage_source,
        "subsystem_count": total_count,
        "covered_subsystem_count": covered_count,
        "coverage_fraction": coverage_fraction,
        "missing_subsystems": missing_subsystems,
        "summary": _coverage_summary_text(covered_count, total_count, missing_subsystems),
        "subsystems": subsystem_reports,
    }


def _default_coverage_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_coverage.json")


def _sample_coverage_key(sample: Dict[str, object]) -> str:
    return "|".join(
        [
            str(sample["record_id"]),
            str(sample["system"]),
            _format_mole_fractions(sample["mole_fractions"]),
        ]
    )


def predict_from_config(
    config_path: Path,
    property_name: str,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
    coverage_output_path: Optional[Path] = None,
) -> Dict[str, object]:
    config = load_config(config_path)
    artifact_dir = Path(config["training"]["output_dir"])
    temperatures_per_record = int(config["data"].get("temperatures_per_record", 1))
    rows = load_json_with_fallback(input_path)
    if not isinstance(rows, list):
        raise ValueError("Prediction input must be a JSON array of raw-data-style records")

    samples = _prediction_rows(rows, property_name, temperatures_per_record)
    dataset = PropertyDataset(artifact_dir, "test")
    encoded_samples = [dataset.encode_sample(sample) for sample in samples]
    torch, _, _ = _lazy_torch()
    spec = get_property_spec(property_name)
    temperature_law = resolve_model_temperature_law(config["model"])
    if temperature_law == "arrhenius" and not spec.supports_log_metrics:
        raise ValueError("use_arrhenius is only supported for log-target properties")
    if temperature_law == "linear" and spec.supports_log_metrics:
        raise ValueError("use_linear is only supported for value-target properties")
    model, device = _load_model(config, artifact_dir, checkpoint_path)
    coverage_samples = _load_coverage_samples(artifact_dir)
    subsystem_index = _build_subsystem_index(coverage_samples)
    coverage_reports: Dict[str, Dict[str, object]] = {}
    coverage_temperatures: Dict[str, set[float]] = {}
    for sample in samples:
        coverage_key = _sample_coverage_key(sample)
        coverage_temperatures.setdefault(coverage_key, set()).add(_round_public_float(float(sample["temperature"])))
        if coverage_key not in coverage_reports:
            coverage_reports[coverage_key] = _build_subsystem_coverage_report(sample, subsystem_index)
    for coverage_key, temperatures in coverage_temperatures.items():
        temperature_points = sorted(temperatures)
        coverage_reports[coverage_key]["prediction_temperature_range_K"] = {
            "min": min(temperature_points),
            "max": max(temperature_points),
        }
        coverage_reports[coverage_key]["prediction_temperature_points_K"] = _limited_values(temperature_points)
    if coverage_output_path is None:
        coverage_output_path = _default_coverage_output_path(output_path)

    output_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for sample, encoded in zip(samples, encoded_samples):
            coverage_report = coverage_reports[_sample_coverage_key(sample)]
            model_inputs = [
                torch.tensor([encoded["token_ids"]], dtype=torch.long, device=device),
                torch.tensor([encoded["mole_fractions"]], dtype=torch.float32, device=device),
                torch.tensor([encoded["temperature"]], dtype=torch.float32, device=device),
                torch.tensor([encoded["component_mask"]], dtype=torch.float32, device=device),
            ]
            if temperature_law != "none":
                model_inputs.append(torch.tensor([encoded["raw_temperature"]], dtype=torch.float32, device=device))
            output = model(*model_inputs).detach().cpu().item()
            if spec.supports_log_metrics:
                prediction_log_value = output
                prediction_value = math.exp(output)
            else:
                prediction_log_value = None
                prediction_value = output
            prediction_value_text = _format_output_float(prediction_value)
            output_rows.append(
                {
                    "record_id": sample["record_id"],
                    "system": sample["system"],
                    "temperature": sample["temperature"],
                    "components": "-".join(sample["components"]),
                    "mole_percent": _format_mole_percent(sample["mole_fractions"]),
                    "prediction_value": prediction_value_text,
                    "prediction_unit": spec.display_unit,
                    "prediction_value_with_unit": _format_value_with_unit(prediction_value, spec.display_unit),
                    "prediction_log_value": _format_output_float(prediction_log_value),
                    "subsystems_covered": coverage_report["covered_subsystem_count"],
                    "subsystems_total": coverage_report["subsystem_count"],
                    "subsystem_coverage_fraction": _format_output_float(coverage_report["coverage_fraction"]),
                    "missing_subsystems": ";".join(coverage_report["missing_subsystems"]),
                    "subsystem_coverage_summary": coverage_report["summary"],
                    "subsystem_coverage_report": str(coverage_output_path),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "system",
        "temperature",
        "components",
        "mole_percent",
        "prediction_value",
        "prediction_unit",
        "prediction_value_with_unit",
        "prediction_log_value",
        "subsystems_covered",
        "subsystems_total",
        "subsystem_coverage_fraction",
        "missing_subsystems",
        "subsystem_coverage_summary",
        "subsystem_coverage_report",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    with coverage_output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "property": property_name,
                "prediction_unit": spec.display_unit,
                "coverage_source": str(artifact_dir / "train_samples.json"),
                "records": list(coverage_reports.values()),
            },
            handle,
            indent=2,
        )
    return {
        "property": property_name,
        "input_rows": len(rows),
        "prediction_rows": len(output_rows),
        "prediction_unit": spec.display_unit,
        "output": str(output_path),
        "coverage_output": str(coverage_output_path),
    }
