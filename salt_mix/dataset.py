from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class PropertyDataset:
    def __init__(self, artifact_dir: Path, split: str):
        with (artifact_dir / f"{split}_samples.json").open("r", encoding="utf-8") as handle:
            self.samples: List[Dict[str, object]] = json.load(handle)
        with (artifact_dir / "vocab.json").open("r", encoding="utf-8") as handle:
            self.vocab: Dict[str, int] = json.load(handle)
        with (artifact_dir / "train_samples.json").open("r", encoding="utf-8") as handle:
            self.train_samples: List[Dict[str, object]] = json.load(handle)
        all_samples = list(self.samples)
        for other_split in ("train", "val", "test"):
            path = artifact_dir / f"{other_split}_samples.json"
            if not path.exists() or other_split == split:
                continue
            with path.open("r", encoding="utf-8") as handle:
                all_samples.extend(json.load(handle))
        self.max_components = max(
            (len(sample.get("component_tokens", [])) for sample in all_samples),
            default=0,
        )
        self.max_tokens = max(
            (
                len(token_list)
                for sample in all_samples
                for token_list in sample.get("component_tokens", [])
            ),
            default=0,
        )
        self.temperature_mean = 0.0
        self.temperature_std = 1.0
        summary_path = artifact_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.temperature_mean = float(summary.get("temperature_mean", 0.0))
            self.temperature_std = float(summary.get("temperature_std", 1.0))
            if self.temperature_std <= 0.0:
                self.temperature_std = 1.0

    def __len__(self) -> int:
        return len(self.samples)

    def encode_sample(self, sample: Dict[str, object]) -> Dict[str, object]:
        pad_id = self.vocab["[PAD]"]
        unk_id = self.vocab["[UNK]"]
        max_components = max(
            self.max_components,
            len(sample.get("component_tokens", [])),
            len(sample.get("mole_fractions", [])),
        )

        token_ids: List[List[int]] = []
        component_mask: List[int] = []
        for tokens in sample["component_tokens"][:max_components]:
            encoded = [self.vocab.get(token, unk_id) for token in tokens[: self.max_tokens]]
            encoded += [pad_id] * (self.max_tokens - len(encoded))
            token_ids.append(encoded)
            component_mask.append(1)
        while len(token_ids) < max_components:
            token_ids.append([pad_id] * self.max_tokens)
            component_mask.append(0)

        mole_fractions = list(sample["mole_fractions"])[:max_components]
        mole_fractions += [0.0] * (max_components - len(mole_fractions))
        training_target_key = "target_log_value" if "target_log_value" in sample else "target_value"

        raw_temperature = float(sample["temperature"])
        encoded_sample = {
            "record_id": sample["record_id"],
            "system": sample["system"],
            "token_ids": token_ids,
            "component_mask": component_mask,
            "mole_fractions": mole_fractions,
            "temperature": (raw_temperature - self.temperature_mean) / self.temperature_std,
            "raw_temperature": raw_temperature,
            "target_value": float(sample["target_value"]),
            "training_target": float(sample[training_target_key]),
        }
        if "target_log_value" in sample:
            encoded_sample["target_log_value"] = float(sample["target_log_value"])
        return encoded_sample


def load_encoded_split(artifact_dir: Path, split: str) -> List[Dict[str, object]]:
    dataset = PropertyDataset(artifact_dir, split)
    return [dataset.encode_sample(sample) for sample in dataset.samples]
