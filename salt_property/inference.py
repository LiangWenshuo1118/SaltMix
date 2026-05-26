from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from .chemistry import classify_component, tokenize_components
from .config_utils import resolve_model_rk_order, resolve_model_temperature_law, resolve_model_use_phy
from .dataset import PropertyDataset
from .preprocess import build_axis_points, load_config, load_json_with_fallback
from .properties import get_property_spec, parse_composition, parse_temperature_point, parse_temperature_range
from .train import _lazy_torch


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


def predict_from_config(
    config_path: Path,
    property_name: str,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Optional[Path] = None,
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

    output_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for sample, encoded in zip(samples, encoded_samples):
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
            output_rows.append(
                {
                    "record_id": sample["record_id"],
                    "system": sample["system"],
                    "temperature": sample["temperature"],
                    "components": "-".join(sample["components"]),
                    "mole_fractions": "-".join(f"{value:.10g}" for value in sample["mole_fractions"]),
                    "prediction_value": prediction_value,
                    "prediction_log_value": prediction_log_value,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_id",
        "system",
        "temperature",
        "components",
        "mole_fractions",
        "prediction_value",
        "prediction_log_value",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return {"property": property_name, "input_rows": len(rows), "prediction_rows": len(output_rows), "output": str(output_path)}
