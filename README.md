# SaltMix

Physics-inspired neural models for molten-salt thermophysical and transport properties. SaltMix trains on unary and binary systems for density, viscosity, surface tension, and electrical conductivity, then evaluates mixture-order extrapolation on ternary and higher-order systems.

## Repository Contents

- `salt_mix/`: Core Python package for preprocessing, chemistry parsing, modeling, training, evaluation, and inference.
- `configs/`: YAML configurations for each property and model variant.
- `data/`: Curated Janz JSON data, the original Janz PDF, and the MSTDB-TP access pointer.
- `tests/`: Unit tests for the CLI, preprocessing, parsing, and model behavior.
- `docs/`: English manuscript, supporting information, selected result documents, figures, and the compiled manuscript PDF.
- `outputs/`: Selected best-model training artifacts and processed result summaries.

## Model Overview

All models share a component-level Transformer encoder. Each salt component is parsed into structured tokens and supplied to the model with mole fraction and temperature.

Supported output structures include:

- `use_phy: true`, `use_rk: true`: composition-weighted prediction with Redlich-Kister-style pair correction.
- `use_phy: true`, `use_rk: false`: composition-weighted prediction.
- `use_phy: false`: direct prediction after Set Transformer pooling.
- `use_linear: true`: linear temperature-constrained models for density and surface tension.
- `use_arrhenius: true`: Arrhenius temperature-constrained models for viscosity and electrical conductivity.

Density and surface tension are trained in the original property space. Viscosity and electrical conductivity are trained in natural-log space, with metrics and predictions reported in both original and log spaces.

## Installation

SaltMix is currently run directly from the repository root with `PYTHONPATH`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

Core dependencies:

- `numpy`
- `pandas`
- `PyYAML`
- `torch`

## Quick Start

Example workflow for density:

```bash
export PYTHONPATH="$PWD"

python -m salt_mix preprocess \
  --config configs/density_config.yaml \
  --property density

python -m salt_mix train \
  --config configs/density_config.yaml \
  --property density

python -m salt_mix eval \
  --config configs/density_config.yaml \
  --property density
```

`train` automatically runs preprocessing first when the required preprocessed files are missing. The default output directory is controlled by `training.output_dir` in each configuration file.

## Properties and Configurations

| Property | Default RK configuration | Temperature-constrained configuration | Default output directory |
| --- | --- | --- | --- |
| `density` | `configs/density_config.yaml` | `configs/density_linear_config.yaml` | `outputs/density_results` |
| `viscosity` | `configs/viscosity_config.yaml` | `configs/viscosity_arrhenius_config.yaml` | `outputs/viscosity_results` |
| `surface_tension` | `configs/surface_tension_config.yaml` | `configs/surface_tension_linear_config.yaml` | `outputs/surface_tension_results` |
| `electrical_conductivity` | `configs/electrical_conductivity_config.yaml` | `configs/electrical_conductivity_arrhenius_config.yaml` | `outputs/electrical_conductivity_results` |

The CLI `--property` value must be one of:

```text
density
viscosity
surface_tension
electrical_conductivity
```

Arrhenius variants still use the base property name:

```bash
python -m salt_mix train \
  --config configs/viscosity_arrhenius_config.yaml \
  --property viscosity
```

## Data Preprocessing

Preprocessing reads raw data directories from the selected configuration, for example:

```yaml
data:
  raw_density_dir: data/density
  validation_ratio: 0.2
  temperatures_per_record: 5
```

The preprocessing pipeline:

- Parses temperature ranges, composition ranges, and property correlation equations.
- Filters organic salts, oxides, sulfides, and systems explicitly excluded by configuration.
- Splits unary and binary systems into training and validation sets.
- Uses ternary and higher-order systems as the mixture-order extrapolation test set.
- Samples record-level data by temperature and composition.
- Builds the component-token vocabulary.

Preprocessed files are written to `training.output_dir`:

- `train_samples.json`
- `val_samples.json`
- `test_samples.json`
- `filter_report.json`
- `vocab.json`
- `summary.json`

## Training and Evaluation Artifacts

Training uses AdamW, exponential moving average weights, and early stopping. Training and evaluation produce:

- `best_model.pt`: best validation EMA checkpoint.
- `epoch_metrics.csv`: per-epoch training and validation metrics.
- `metrics.json`: validation and ternary-or-higher test metrics.
- `val_predictions.csv`: validation-set predictions.
- `test_predictions.csv`: test-set predictions.

For viscosity and electrical conductivity, `metrics.json` includes both original-value and log-space metrics.
Prediction CSVs include both separate unit columns and `*_with_unit` columns for reported property values.
`epoch_metrics.csv` includes `property_unit` so RMSE/MAE columns can be interpreted in the original property scale
when they are not log-space metrics.

The repository includes curated Janz JSON data and `data/Janz_1988/Janz.pdf`. MSTDB-TP CSV/JSON data files are not redistributed here; `data/mstdb-tp-v4/README.md` points to the official ORNL access page at https://msd.ornl.gov/.

## Inference

Inference input is a raw-data-style JSON array. The `equation` and `unit` fields may be empty, while `system`, `composition`, and `temperature` are required.

Example input:

```json
[
  {
    "index": "PRED_1",
    "data": {
      "system": "MgCl2-CaCl2-NaCl-KCl",
      "composition": "4.96-3.27-51.44-40.33 mol%",
      "equation": "",
      "temperature": "973 K",
      "unit": ""
    }
  }
]
```

Run prediction:

```bash
python -m salt_mix predict \
  --config configs/viscosity_config.yaml \
  --property viscosity \
  --input path/to/predict_input.json \
  --output outputs/predict/viscosity_predictions.csv \
  --coverage-output outputs/predict/viscosity_predictions_coverage.json
```

When `--checkpoint` is omitted, prediction reads `best_model.pt` from the configured output directory.
Prediction CSVs include a lower-order subsystem coverage summary. A detailed JSON coverage report is also written to
`--coverage-output`; when omitted, SaltMix writes `<output-stem>_coverage.json` next to the prediction CSV. The report
lists covered/missing lower-order subsystems with observed composition ranges, composition points, temperature ranges,
temperature points, source systems, record IDs, and the query composition projected onto each subsystem from the model
artifact's `train_samples.json`.
Prediction outputs report compositions as `mole_percent`; predicted property/log values and coverage scores are rounded
to three decimal places. Property CSVs report units explicitly, including `prediction_unit` and
`prediction_value_with_unit`. Model output units are `g/cm^3` for density, `mPa s` for viscosity, `mN/m` for surface
tension, and `S/cm` for electrical conductivity. Janz electrical-conductivity records written as `ohm^-1 cm^-1` are
equivalent to `S/cm`; SaltMix does not apply an `S/m` conversion during electrical-conductivity inference.

## Testing

```bash
export PYTHONPATH="$PWD"
python3 -m unittest discover tests
```

Tests cover:

- CLI dispatch.
- Temperature, composition, and correlation-equation parsing.
- Component chemical-formula tokenization.
- RK, linear, Arrhenius, and no-physics model branches.
- Prediction output fields and log-space metrics.

## Documents

- [docs/Salt_en.md](docs/Salt_en.md): English manuscript draft.
- [docs/Salt_SI.md](docs/Salt_SI.md): English supporting-information draft.
- [docs/jctc-latex-template/manuscript_en.pdf](docs/jctc-latex-template/manuscript_en.pdf): English manuscript PDF.
- [docs/jctc-latex-template/manuscript_en.tex](docs/jctc-latex-template/manuscript_en.tex): English manuscript LaTeX source.
- [docs/Janz-data-corrections.md](docs/Janz-data-corrections.md): Janz data-correction record.
