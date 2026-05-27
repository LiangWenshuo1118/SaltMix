from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .config_utils import resolve_model_rk_order, resolve_model_temperature_law, resolve_model_use_phy
from .dataset import load_encoded_split
from .preprocess import load_config, preprocess_from_config
from .properties import get_property_spec


def _lazy_torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    return torch, nn, DataLoader


def build_dataloader(encoded_samples: List[Dict[str, object]], batch_size: int, shuffle: bool):
    torch, _, DataLoader = _lazy_torch()

    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        payload = {
            "record_id": [item["record_id"] for item in batch],
            "system": [item["system"] for item in batch],
            "token_ids": torch.tensor([item["token_ids"] for item in batch], dtype=torch.long),
            "component_mask": torch.tensor([item["component_mask"] for item in batch], dtype=torch.float32),
            "mole_fractions": torch.tensor([item["mole_fractions"] for item in batch], dtype=torch.float32),
            "temperature": torch.tensor([item["temperature"] for item in batch], dtype=torch.float32),
            "raw_temperature": torch.tensor(
                [item.get("raw_temperature", item["temperature"]) for item in batch],
                dtype=torch.float32,
            ),
            "target_value": torch.tensor([item["target_value"] for item in batch], dtype=torch.float32),
            "training_target": torch.tensor([item["training_target"] for item in batch], dtype=torch.float32),
        }
        if all("target_log_value" in item for item in batch):
            payload["target_log_value"] = torch.tensor([item["target_log_value"] for item in batch], dtype=torch.float32)
        return payload

    return DataLoader(encoded_samples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def model_forward(model, batch: Dict[str, object], device):
    raw_temperature = batch.get("raw_temperature")
    if getattr(model, "temperature_law", "none") != "none" and raw_temperature is not None:
        return model(
            batch["token_ids"].to(device),
            batch["mole_fractions"].to(device),
            batch["temperature"].to(device),
            batch["component_mask"].to(device),
            raw_temperature.to(device),
        )
    return model(
        batch["token_ids"].to(device),
        batch["mole_fractions"].to(device),
        batch["temperature"].to(device),
        batch["component_mask"].to(device),
    )


def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"rmse": 0.0, "mae": 0.0, "pearson_r": 0.0}
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    mean_true = sum(y_true) / n
    mean_pred = sum(y_pred) / n
    true_variance = sum((value - mean_true) ** 2 for value in y_true)
    pred_variance = sum((value - mean_pred) ** 2 for value in y_pred)
    covariance = sum((true_value - mean_true) * (pred_value - mean_pred) for true_value, pred_value in zip(y_true, y_pred))
    pearson_r = covariance / math.sqrt(true_variance * pred_variance) if true_variance and pred_variance else 0.0
    return {"rmse": math.sqrt(mse), "mae": mae, "pearson_r": pearson_r}


def _public_float(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return numeric
    return float(f"{numeric:.8g}")


def _public_value_with_unit(value: float, unit: str) -> str:
    return f"{_public_float(value)} {unit}"


def build_loss_fn(loss_name: str):
    torch, nn, _ = _lazy_torch()
    normalized = loss_name.strip().lower()
    if normalized == "mse":
        return nn.MSELoss()
    if normalized == "log_cosh":
        def log_cosh_loss(preds, targets):
            return torch.mean(torch.log(torch.cosh(preds - targets + 1e-12)))

        return log_cosh_loss
    raise ValueError(f"Unsupported loss: {loss_name}")


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    torch, _, _ = _lazy_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_history_fieldnames(property_name: str) -> List[str]:
    spec = get_property_spec(property_name)
    if spec.supports_log_metrics:
        return ["epoch", "train_loss", "val_loss", "val_rmse_log", "val_mae_log", "property_unit"]
    return ["epoch", "train_loss", "val_loss", "val_rmse", "val_mae", "property_unit"]


def write_epoch_history(
    output_dir: Path,
    history: List[Dict[str, float | int | None]],
    property_name: str,
) -> None:
    spec = get_property_spec(property_name)
    rows = [
        {
            **row,
            "property_unit": spec.display_unit,
        }
        for row in history
    ]
    csv_path = output_dir / "epoch_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=epoch_history_fieldnames(property_name),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(csv_safe_rows(rows))


def evaluate_model(model, dataloader, device, property_name: str) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    torch, _, _ = _lazy_torch()
    spec = get_property_spec(property_name)
    model.eval()
    predictions: List[Dict[str, object]] = []
    y_true_value: List[float] = []
    y_pred_value: List[float] = []
    y_true_log: List[float] = []
    y_pred_log: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model_forward(model, batch, device)
            raw_outputs = outputs.detach().cpu().tolist()
            if spec.supports_log_metrics:
                pred_log_values = raw_outputs
                pred_values = [math.exp(value) for value in pred_log_values]
                target_log_values = batch["target_log_value"].tolist()
                target_values = batch["target_value"].tolist()
                for record_id, system, pred_log, pred_value, target_log, target_value in zip(
                    batch["record_id"],
                    batch["system"],
                    pred_log_values,
                    pred_values,
                    target_log_values,
                    target_values,
                ):
                    public_pred_log = _public_float(pred_log)
                    public_prediction_value = _public_float(math.exp(public_pred_log))
                    public_target_value = _public_float(target_value)
                    predictions.append(
                        {
                            "record_id": record_id,
                            "system": system,
                            "prediction_value": public_prediction_value,
                            "prediction_unit": spec.display_unit,
                            "prediction_value_with_unit": _public_value_with_unit(
                                public_prediction_value,
                                spec.display_unit,
                            ),
                            "target_value": public_target_value,
                            "target_unit": spec.display_unit,
                            "target_value_with_unit": _public_value_with_unit(
                                public_target_value,
                                spec.display_unit,
                            ),
                            "prediction_log_value": public_pred_log,
                            "target_log_value": _public_float(target_log),
                        }
                    )
                y_true_log.extend(target_log_values)
                y_pred_log.extend(pred_log_values)
                y_true_value.extend(target_values)
                y_pred_value.extend(pred_values)
            else:
                target_values = batch["target_value"].tolist()
                for record_id, system, pred_value, target_value in zip(
                    batch["record_id"], batch["system"], raw_outputs, target_values
                ):
                    public_prediction_value = _public_float(pred_value)
                    public_target_value = _public_float(target_value)
                    predictions.append(
                        {
                            "record_id": record_id,
                            "system": system,
                            "prediction_value": public_prediction_value,
                            "prediction_unit": spec.display_unit,
                            "prediction_value_with_unit": _public_value_with_unit(
                                public_prediction_value,
                                spec.display_unit,
                            ),
                            "target_value": public_target_value,
                            "target_unit": spec.display_unit,
                            "target_value_with_unit": _public_value_with_unit(
                                public_target_value,
                                spec.display_unit,
                            ),
                            "prediction_log_value": None,
                            "target_log_value": None,
                        }
                    )
                y_true_value.extend(target_values)
                y_pred_value.extend(raw_outputs)

    metrics = compute_metrics(y_true_value, y_pred_value)
    metrics.update({"rmse_log": None, "mae_log": None, "pearson_r_log": None})
    if spec.supports_log_metrics:
        log_metrics = compute_metrics(y_true_log, y_pred_log)
        metrics.update(
            {
                "rmse_log": log_metrics["rmse"],
                "mae_log": log_metrics["mae"],
                "pearson_r_log": log_metrics["pearson_r"],
            }
        )
    return metrics, predictions


def evaluate_loss(model, dataloader, device, loss_fn) -> float:
    torch, _, _ = _lazy_torch()
    model.eval()
    loss_total = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            preds = model_forward(model, batch, device)
            loss = loss_fn(preds, batch["training_target"].to(device))
            loss_total += float(loss.detach().cpu())
            batch_count += 1
    return loss_total / max(batch_count, 1)


def csv_safe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    safe_rows: List[Dict[str, object]] = []
    for row in rows:
        safe_rows.append({key: ("nan" if value is None else value) for key, value in row.items()})
    return safe_rows


def clone_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def update_ema_state(ema_state, model, decay: float) -> None:
    current_state = model.state_dict()
    for key, value in current_state.items():
        ema_state[key].mul_(decay).add_(value.detach().cpu(), alpha=1.0 - decay)


def evaluate_with_state(model, state_dict, dataloader, device, property_name: str, loss_fn):
    original_state = clone_state_dict(model.state_dict())
    model.load_state_dict(state_dict)
    try:
        val_loss = evaluate_loss(model, dataloader, device, loss_fn)
        val_metrics, predictions = evaluate_model(model, dataloader, device, property_name)
    finally:
        model.load_state_dict(original_state)
    return val_loss, val_metrics, predictions


def train_from_config(config_path: Path, property_name: str) -> Dict[str, object]:
    config = load_config(config_path)
    set_training_seed(int(config["training"]["seed"]))
    artifact_dir = Path(config["training"]["output_dir"])
    if not (artifact_dir / "train_samples.json").exists():
        preprocess_from_config(config_path, property_name)
    spec = get_property_spec(property_name)

    torch, nn, _ = _lazy_torch()
    from .model import PropertyRKModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples = load_encoded_split(artifact_dir, "train")
    val_samples = load_encoded_split(artifact_dir, "val")
    with (artifact_dir / "vocab.json").open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    temperature_law = resolve_model_temperature_law(config["model"])
    if temperature_law == "arrhenius" and not spec.supports_log_metrics:
        raise ValueError("use_arrhenius is only supported for log-target properties")
    if temperature_law == "linear" and spec.supports_log_metrics:
        raise ValueError("use_linear is only supported for value-target properties")
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
        max_components=len(train_samples[0]["token_ids"]) if train_samples else None,
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    train_loader = build_dataloader(train_samples, int(config["training"]["batch_size"]), shuffle=True)
    val_loader = build_dataloader(val_samples, int(config["training"]["batch_size"]), shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    loss_fn = build_loss_fn(str(config["training"].get("loss", "mse")))
    ema_decay = float(config["training"].get("ema_decay", 0.98))
    early_stopping_patience = int(config["training"].get("early_stopping_patience", 10))
    grad_clip_max_norm = config["training"].get("grad_clip_max_norm")
    grad_clip_max_norm = None if grad_clip_max_norm is None else float(grad_clip_max_norm)
    ema_state = clone_state_dict(model.state_dict())

    best_val_metric = float("inf")
    epochs_without_improvement = 0
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "best_model.pt"
    epoch_history: List[Dict[str, float | int | None]] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model_forward(model, batch, device)
            loss = loss_fn(preds, batch["training_target"].to(device))
            loss.backward()
            if grad_clip_max_norm is not None and grad_clip_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
            update_ema_state(ema_state, model, ema_decay)
            train_loss_total += float(loss.detach().cpu())
            train_batches += 1
        val_loss, val_metrics, _ = evaluate_with_state(model, ema_state, val_loader, device, property_name, loss_fn)
        train_loss = train_loss_total / max(train_batches, 1)
        epoch_row: Dict[str, float | int | None] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if spec.supports_log_metrics:
            epoch_row.update(
                {
                    "val_rmse_log": val_metrics.get("rmse_log"),
                    "val_mae_log": val_metrics.get("mae_log"),
                }
            )
        else:
            epoch_row.update({"val_rmse": val_metrics["rmse"], "val_mae": val_metrics["mae"]})
        epoch_history.append(epoch_row)
        write_epoch_history(artifact_dir, epoch_history, property_name)
        print(epoch_row)
        objective_metric = val_metrics["rmse_log"] if spec.supports_log_metrics else val_metrics["rmse"]
        if objective_metric < best_val_metric:
            best_val_metric = objective_metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": clone_state_dict(ema_state),
                    "config": config,
                    "property": property_name,
                    "best_metric": best_val_metric,
                    "best_epoch": epoch,
                    "ema_decay": ema_decay,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    with checkpoint_path.open("rb"):
        pass
    return evaluate_from_config(config_path, property_name, checkpoint_path)


def evaluate_from_config(config_path: Path, property_name: str, checkpoint_path: Path | None = None) -> Dict[str, object]:
    config = load_config(config_path)
    artifact_dir = Path(config["training"]["output_dir"])
    torch, _, _ = _lazy_torch()
    from .model import PropertyRKModel

    with (artifact_dir / "vocab.json").open("r", encoding="utf-8") as handle:
        vocab = json.load(handle)

    if checkpoint_path is None:
        checkpoint_path = artifact_dir / "best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_samples = load_encoded_split(artifact_dir, "train")
    spec = get_property_spec(property_name)
    temperature_law = resolve_model_temperature_law(config["model"])
    if temperature_law == "arrhenius" and not spec.supports_log_metrics:
        raise ValueError("use_arrhenius is only supported for log-target properties")
    if temperature_law == "linear" and spec.supports_log_metrics:
        raise ValueError("use_linear is only supported for value-target properties")
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
        max_components=len(train_samples[0]["token_ids"]) if train_samples else None,
        dropout=float(config["model"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_loader = build_dataloader(load_encoded_split(artifact_dir, "val"), int(config["training"]["batch_size"]), shuffle=False)
    test_loader = build_dataloader(load_encoded_split(artifact_dir, "test"), int(config["training"]["batch_size"]), shuffle=False)
    val_metrics, val_predictions = evaluate_model(model, val_loader, device, property_name)
    test_metrics, test_predictions = evaluate_model(model, test_loader, device, property_name)

    output_dir = artifact_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {"property": property_name, "validation": val_metrics, "ternary_test": test_metrics}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    fieldnames = [
        "record_id",
        "system",
        "prediction_value",
        "prediction_unit",
        "prediction_value_with_unit",
        "target_value",
        "target_unit",
        "target_value_with_unit",
        "prediction_log_value",
        "target_log_value",
    ]
    for filename, rows in (("val_predictions.csv", val_predictions), ("test_predictions.csv", test_predictions)):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_safe_rows(rows))
    return metrics_payload
