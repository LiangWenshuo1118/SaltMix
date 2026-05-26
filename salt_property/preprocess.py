from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .chemistry import classify_component, tokenize_components
from .config_utils import normalize_ratio, resolve_low_data_learning
from .properties import (
    evaluate_property_targets,
    get_property_spec,
    parse_composition,
    parse_property_equation,
    parse_temperature_point,
    parse_temperature_range,
)

COMPOSITION_RANGE_WITH_LABEL_PATTERNS = [
    re.compile(
        r"^\s*(?P<lo>\d+(?:\.\d+)?)\s*-\s*(?P<hi>\d+(?:\.\d+)?)\s*mol%\s*(?P<label>.+?)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?P<lo>\d+(?:\.\d+)?)\s*-\s*(?P<hi>\d+(?:\.\d+)?)\s*(?P<label>.+?)\s*mol%\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?P<lo>\d+(?:\.\d+)?)\s*-\s*(?P<hi>\d+(?:\.\d+)?)\s+(?P<label>.+?)\s*$",
        re.IGNORECASE,
    ),
]


@dataclass
class CompositionSpec:
    mode: str
    mole_fractions: Optional[List[float]] = None
    component_index: Optional[int] = None
    composition_min: Optional[float] = None
    composition_max: Optional[float] = None


@dataclass
class TemperatureSpec:
    mode: str
    minimum: float
    maximum: float


@dataclass
class NormalizedRecord:
    record_id: str
    system: str
    components: List[str]
    composition_spec: CompositionSpec
    temperature_spec: TemperatureSpec
    equation_params: Dict[str, object]
    unit: str
    parser_warning: bool
    tokenized_components: List[Dict[str, object]]


def parse_temperature_spec(raw: str, equation_params: Dict[str, object]) -> TemperatureSpec:
    try:
        lo, hi = parse_temperature_range(raw)
        return TemperatureSpec(mode="range", minimum=lo, maximum=hi)
    except ValueError:
        kind = str(equation_params.get("kind", ""))
        if kind not in {"constant", "composition_poly"}:
            raise
        point = parse_temperature_point(raw)
        return TemperatureSpec(mode="point", minimum=point, maximum=point)


def parse_composition_spec(raw: str, components: Sequence[str], equation_params: Dict[str, object]) -> CompositionSpec:
    try:
        return CompositionSpec(mode="exact", mole_fractions=parse_composition(raw, len(components)))
    except ValueError:
        kind = str(equation_params.get("kind", ""))
        if kind != "composition_poly" or len(components) != 2:
            raise
    match = None
    for pattern in COMPOSITION_RANGE_WITH_LABEL_PATTERNS:
        match = pattern.match(raw)
        if match:
            break
    if not match:
        raise ValueError(f"Unsupported composition range: {raw}")
    label = match.group("label").strip()
    try:
        component_index = next(index for index, component in enumerate(components) if component == label)
    except StopIteration as exc:
        raise ValueError(f"Composition label does not match system components: {raw}") from exc
    lo = float(match.group("lo")) / 100.0
    hi = float(match.group("hi")) / 100.0
    return CompositionSpec(
        mode="range",
        component_index=component_index,
        composition_min=min(lo, hi),
        composition_max=max(lo, hi),
    )


def build_axis_points(lower: float, upper: float, count: int) -> List[float]:
    if count <= 1 or abs(upper - lower) < 1e-12:
        return [0.5 * (lower + upper)]
    step = (upper - lower) / (count - 1)
    return [lower + step * index for index in range(count)]


def build_interval_points(lower: float, upper: float, interval: float) -> List[float]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    if abs(upper - lower) < 1e-12:
        return [lower]

    points: List[float] = []
    index = 0
    while True:
        point = min(lower + interval * index, upper)
        if not points or abs(point - points[-1]) > 1e-12:
            points.append(point)
        if point >= upper - 1e-12:
            break
        index += 1
    return points


def build_mole_fraction_points(
    record: NormalizedRecord,
    compositions_per_record: int,
    composition_interval: Optional[float] = None,
) -> List[Tuple[List[float], Optional[float]]]:
    spec = record.composition_spec
    if spec.mode == "exact":
        assert spec.mole_fractions is not None
        return [(spec.mole_fractions, None)]

    assert spec.component_index is not None
    assert spec.composition_min is not None
    assert spec.composition_max is not None

    if composition_interval is not None:
        points = build_interval_points(
            spec.composition_min,
            spec.composition_max,
            composition_interval / 100.0,
        )
    else:
        points = build_axis_points(spec.composition_min, spec.composition_max, compositions_per_record)
    results: List[Tuple[List[float], Optional[float]]] = []
    for point in points:
        mole_fractions = [0.0] * len(record.components)
        mole_fractions[spec.component_index] = point
        complement_index = 1 - spec.component_index
        mole_fractions[complement_index] = 1.0 - point
        results.append((mole_fractions, point * 100.0))
    return results


def build_random_point_split(
    samples: Sequence[Dict[str, object]], validation_ratio: float, seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * validation_ratio))) if shuffled else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    if not train_samples and val_samples:
        train_samples.append(val_samples.pop())
    return train_samples, val_samples


def build_random_record_split(
    records: Sequence[NormalizedRecord], validation_ratio: float, seed: int
) -> Tuple[List[NormalizedRecord], List[NormalizedRecord]]:
    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * validation_ratio))) if shuffled else 0
    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    if not train_records and val_records:
        train_records.append(val_records.pop())
    return train_records, val_records


def apply_low_data_learning(
    samples: Sequence[Dict[str, object]], enabled: bool, learning_ratio: float
) -> List[Dict[str, object]]:
    selected = list(samples)
    if not enabled:
        return selected
    selected_count = max(1, int(round(len(selected) * learning_ratio))) if selected else 0
    return selected[:selected_count]


def compute_temperature_stats(samples: Sequence[Dict[str, object]]) -> Tuple[float, float]:
    temperatures = [float(sample["temperature"]) for sample in samples]
    if not temperatures:
        return 0.0, 1.0
    mean = sum(temperatures) / len(temperatures)
    variance = sum((temperature - mean) ** 2 for temperature in temperatures) / len(temperatures)
    std = variance ** 0.5
    return mean, std if std > 0.0 else 1.0


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json_with_fallback(path: Path):
    for encoding in ("utf-8", "latin-1"):
        try:
            with path.open("r", encoding=encoding) as handle:
                return json.load(handle)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Unable to decode JSON file: {path}")


def load_raw_rows(data_cfg: Dict[str, object], property_name: str) -> List[Dict[str, object]]:
    raw_dir_key = get_property_spec(property_name).raw_data_key
    raw_dir_value = data_cfg.get(raw_dir_key)
    if raw_dir_value:
        raw_dir = Path(raw_dir_value)
        if property_name == "density":
            rows: List[Dict[str, object]] = []
            for path in sorted(raw_dir.glob("*.json")):
                if path.name.startswith("."):
                    continue
                rows.extend(load_json_with_fallback(path))
            return rows

        rows_by_id: Dict[str, Dict[str, object]] = {}
        for path in sorted(raw_dir.glob("*.json")):
            if path.name.startswith("."):
                continue
            for row in load_json_with_fallback(path):
                rows_by_id[str(row.get("index", ""))] = row
        return list(rows_by_id.values())

    raise ValueError(f"No raw data source configured for property: {property_name}")


def normalize_record(row: Dict[str, object], property_name: str) -> NormalizedRecord:
    data = row["data"]
    record_id = row["index"]
    system = str(data["system"]).strip()
    components = [component.strip() for component in system.split("-")]

    for component in components:
        reason = classify_component(component)
        if reason:
            raise ValueError(reason)

    equation_params = parse_property_equation(property_name, str(data["equation"]))
    temperature_spec = parse_temperature_spec(str(data["temperature"]), equation_params)
    composition_spec = parse_composition_spec(str(data["composition"]), components, equation_params)
    tokenized = tokenize_components(components)
    parser_warning = any(item.parser_warning for item in tokenized)
    return NormalizedRecord(
        record_id=str(record_id),
        system=system,
        components=components,
        composition_spec=composition_spec,
        temperature_spec=temperature_spec,
        equation_params=equation_params,
        unit=str(data["unit"]),
        parser_warning=parser_warning,
        tokenized_components=[
            {
                "formula": item.formula,
                "tokens": item.tokens,
                "parser_warning": item.parser_warning,
                "metadata": item.metadata,
            }
            for item in tokenized
        ],
    )


def expand_record(
    record: NormalizedRecord,
    temperatures_per_record: int,
    property_name: str,
    compositions_per_record: int,
    temperature_interval: Optional[float] = None,
    composition_interval: Optional[float] = None,
) -> List[Dict[str, object]]:
    if record.temperature_spec.mode == "point":
        temperatures = [record.temperature_spec.minimum]
    elif temperature_interval is not None:
        temperatures = build_interval_points(
            record.temperature_spec.minimum,
            record.temperature_spec.maximum,
            temperature_interval,
        )
    else:
        temperatures = build_axis_points(
            record.temperature_spec.minimum,
            record.temperature_spec.maximum,
            temperatures_per_record,
        )
    composition_points = build_mole_fraction_points(
        record,
        compositions_per_record,
        composition_interval=composition_interval,
    )

    samples = []
    for temperature in temperatures:
        for mole_fractions, composition_value in composition_points:
            try:
                target_value, target_log_value = evaluate_property_targets(
                    property_name,
                    record.equation_params,
                    temperature,
                    composition_value=composition_value,
                )
            except ValueError as exc:
                if "Property value must be positive for log metrics" in str(exc):
                    continue
                raise
            sample = {
                "record_id": record.record_id,
                "system": record.system,
                "num_components": len(record.components),
                "components": record.components,
                "component_tokens": [item["tokens"] for item in record.tokenized_components],
                "component_parser_warnings": [item["parser_warning"] for item in record.tokenized_components],
                "mole_fractions": mole_fractions,
                "temperature": temperature,
                "target_value": target_value,
                "unit": record.unit,
            }
            if target_log_value is not None:
                sample["target_log_value"] = target_log_value
            for key, value in record.equation_params.items():
                if isinstance(value, (int, float)):
                    sample[f"equation_{key}"] = value
            samples.append(sample)
    if not samples:
        raise ValueError("no_valid_samples")
    return samples


def build_vocab(samples: Iterable[Dict[str, object]]) -> Dict[str, int]:
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for sample in samples:
        for token_list in sample["component_tokens"]:
            for token in token_list:
                if token not in vocab:
                    vocab[token] = len(vocab)
    return vocab


def preprocess_from_config(config_path: Path, property_name: str) -> Dict[str, object]:
    config = load_config(config_path)
    data_cfg = config["data"]
    training_cfg = config["training"]
    artifact_dir = Path(training_cfg["output_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    excluded_systems = {str(system).strip() for system in data_cfg.get("excluded_systems", [])}
    excluded_record_ids = {str(record_id).strip() for record_id in data_cfg.get("excluded_record_ids", [])}
    temperatures_per_record = int(data_cfg["temperatures_per_record"])
    compositions_per_record = int(data_cfg.get("compositions_per_record", temperatures_per_record))
    raw_temperature_interval = data_cfg.get("temperature_interval", data_cfg.get("temperature_interval_k"))
    temperature_interval = float(raw_temperature_interval) if raw_temperature_interval is not None else None
    raw_composition_interval = data_cfg.get("composition_interval")
    composition_interval = float(raw_composition_interval) if raw_composition_interval is not None else None
    validation_ratio = normalize_ratio(data_cfg["validation_ratio"], field_name="validation_ratio")
    low_data_learning, low_data_learning_ratio = resolve_low_data_learning(training_cfg.get("low_data_learning", False))

    normalized_records: List[NormalizedRecord] = []
    filter_report: List[Dict[str, str]] = []

    for row in load_raw_rows(data_cfg, property_name):
        try:
            record_id = str(row.get("index", "")).strip()
            if record_id in excluded_record_ids:
                raise ValueError("excluded_record")
            system = str(row.get("data", {}).get("system", "")).strip()
            if system in excluded_systems:
                raise ValueError("excluded_system")
            normalized_records.append(normalize_record(row, property_name))
        except Exception as exc:
            filter_report.append(
                {
                    "record_id": str(row.get("index", "")),
                    "system": str(row.get("data", {}).get("system", "")),
                    "reason": str(exc),
                }
            )

    train_val_records: List[NormalizedRecord] = []
    test_records: List[NormalizedRecord] = []
    for record in normalized_records:
        if len(record.components) <= 2:
            train_val_records.append(record)
        elif len(record.components) >= 3:
            test_records.append(record)

    train_records, val_records = build_random_record_split(
        train_val_records,
        validation_ratio,
        int(training_cfg["seed"]),
    )

    def expand_records(records: Sequence[NormalizedRecord]) -> List[Dict[str, object]]:
        samples: List[Dict[str, object]] = []
        for record in records:
            try:
                samples.extend(
                    expand_record(
                        record,
                        temperatures_per_record,
                        property_name,
                        compositions_per_record,
                        temperature_interval=temperature_interval,
                        composition_interval=composition_interval,
                    )
                )
            except Exception as exc:
                filter_report.append(
                    {
                        "record_id": record.record_id,
                        "system": record.system,
                        "reason": str(exc),
                    }
                )
        return samples

    train_samples = expand_records(train_records)
    val_samples = expand_records(val_records)
    test_samples = expand_records(test_records)

    full_train_sample_count = len(train_samples)
    train_samples = apply_low_data_learning(train_samples, low_data_learning, low_data_learning_ratio)
    temperature_mean, temperature_std = compute_temperature_stats(train_samples)
    train_systems = set(sample["system"] for sample in train_samples)
    val_systems = set(sample["system"] for sample in val_samples)

    vocab = build_vocab(train_samples + val_samples + test_samples)
    outputs = {
        "train_samples.json": train_samples,
        "val_samples.json": val_samples,
        "test_samples.json": test_samples,
        "filter_report.json": filter_report,
        "vocab.json": vocab,
        "summary.json": {
            "property": property_name,
            "normalized_records": len(normalized_records),
            "filtered_records": len(filter_report),
            "excluded_systems": sorted(excluded_systems),
            "excluded_record_ids": sorted(excluded_record_ids),
            "validation_ratio": validation_ratio,
            "temperature_interval": temperature_interval,
            "temperature_mean": temperature_mean,
            "temperature_std": temperature_std,
            "composition_interval": composition_interval,
            "low_data_learning": low_data_learning,
            "low_data_learning_ratio": low_data_learning_ratio,
            "train_records": len(train_records),
            "val_records": len(val_records),
            "test_records": len(test_records),
            "train_samples_before_low_data": full_train_sample_count,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "systems_train": len(train_systems),
            "systems_val": len(val_systems),
        },
    }
    for name, payload in outputs.items():
        with (artifact_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return outputs["summary.json"]
