#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <property> <run_dir> [archive_name] [experiment_time]"
  exit 1
fi

PROPERTY="$1"
RUN_DIR="$2"
ARCHIVE_NAME="${3:-}"
EXPERIMENT_TIME="${4:-$(date +%Y_%m_%d_%H_%M)}"
TRAINING_PCT="${TRAINING_PCT:-20}"

resolve_config_path() {
  case "$1" in
    density)
      echo "$REPO_ROOT/configs/density_config.yaml"
      ;;
    viscosity)
      echo "$REPO_ROOT/configs/viscosity_config.yaml"
      ;;
    viscosity_arrhenius)
      echo "$REPO_ROOT/configs/viscosity_arrhenius_config.yaml"
      ;;
    surface_tension)
      echo "$REPO_ROOT/configs/surface_tension_config.yaml"
      ;;
    electrical_conductivity)
      echo "$REPO_ROOT/configs/electrical_conductivity_config.yaml"
      ;;
    electrical_conductivity_arrhenius)
      echo "$REPO_ROOT/configs/electrical_conductivity_arrhenius_config.yaml"
      ;;
    *)
      return 1
      ;;
  esac
}

cli_property_name() {
  case "$1" in
    viscosity_arrhenius)
      echo "viscosity"
      ;;
    electrical_conductivity_arrhenius)
      echo "electrical_conductivity"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

normalize_n_label() {
  local rk_order="$1"
  if [[ "$rk_order" == "None" || "$rk_order" == "none" ]]; then
    printf 'none\n'
  else
    printf '%s\n' "$rk_order"
  fi
}

model_setting_slug() {
  local training_pct="$1"
  local use_phy="$2"
  local use_rk="$3"
  local rk_order="$4"
  local n_label
  n_label="$(normalize_n_label "$rk_order")"
  printf 'training_%spct_phys_%s_rk_%s_n_%s\n' "$training_pct" "$use_phy" "$use_rk" "$n_label"
}

formula_model_setting_slug() {
  local use_phy="$1"
  local use_rk="$2"
  local use_arrhenius="$3"
  local rk_order="$4"
  local n_label
  if [[ "$use_phy" == "false" ]]; then
    printf 'phys_false\n'
    return
  fi
  if [[ "$use_arrhenius" == "true" ]]; then
    printf 'phys_true_arrhenius\n'
    return
  fi
  if [[ "$use_rk" == "false" ]]; then
    printf 'phys_true_rk_false\n'
    return
  fi
  n_label="$(normalize_n_label "$rk_order")"
  printf 'phys_true_rk_true_n_%s\n' "$n_label"
}

linear_model_setting_slug() {
  local use_phy="$1"
  local use_linear="$2"
  if [[ "$use_phy" == "true" && "$use_linear" == "true" ]]; then
    printf 'phys_true_linear\n'
    return
  fi
  echo "unsupported linear model setting: use_phy=$use_phy use_linear=$use_linear" >&2
  return 1
}

run_fractional_serial() {
  local training_pct="$1"
  shift
  local properties=(density viscosity surface_tension electrical_conductivity)
  local seeds=(42 43 44 45 46)
  local model_specs=("$@")

  for property_name in "${properties[@]}"; do
    for spec in "${model_specs[@]}"; do
      read -r use_phy use_rk rk_order <<<"$spec"
      model_setting="$(model_setting_slug "$training_pct" "$use_phy" "$use_rk" "$rk_order")"
      model_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}"
      mkdir -p "$model_dir"
      for seed in "${seeds[@]}"; do
        work_dir="$model_dir/seed=${seed}"
        mkdir -p "$work_dir"
        config_path="$work_dir/config.yaml"
        write_model_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path" "$training_pct" "$use_phy" "$use_rk" "$rk_order"
        printf 'starting %s low-data=%spct phys=%s rk=%s n=%s seed=%s\n' \
          "$property_name" "$training_pct" "$use_phy" "$use_rk" "$(normalize_n_label "$rk_order")" "$seed" >"$work_dir/run.log"
        (
          cd "$work_dir"
          PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
          PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
          PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
        ) >>"$work_dir/run.log" 2>&1
      done
    done
  done
}

run_with_config() {
  local property_name="$1"
  local config_path="$2"
  local work_dir="$3"
  mkdir -p "$work_dir"
  cd "$work_dir"
  PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
  PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
  PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
}

use_full_data_no_split() {
  local work_dir="$1"
  python - "$work_dir" <<'PY'
import json
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
split_names = ("train", "val", "test")
splits = {}
for split in split_names:
    path = work_dir / f"{split}_samples.json"
    with path.open(encoding="utf-8") as handle:
        splits[split] = json.load(handle)

all_samples = []
seen = set()
for split in split_names:
    for sample in splits[split]:
        key = (
            sample.get("record_id"),
            sample.get("system"),
            tuple(sample.get("components", [])),
            tuple(sample.get("mole_fractions", [])),
            sample.get("temperature"),
        )
        if key in seen:
            continue
        seen.add(key)
        all_samples.append(sample)

for split in split_names:
    with (work_dir / f"{split}_samples.json").open("w", encoding="utf-8") as handle:
        json.dump(all_samples, handle, indent=2)

summary_path = work_dir / "summary.json"
with summary_path.open(encoding="utf-8") as handle:
    summary = json.load(handle)
summary["full_data_no_split"] = True
summary["original_split_samples"] = {split: len(splits[split]) for split in split_names}
summary["train_samples"] = len(all_samples)
summary["val_samples"] = len(all_samples)
summary["test_samples"] = len(all_samples)
summary["train_samples_before_low_data"] = len(all_samples)
summary["train_records"] = len({sample.get("record_id") for sample in all_samples})
summary["val_records"] = summary["train_records"]
summary["test_records"] = summary["train_records"]
summary["systems_train"] = len({sample.get("system") for sample in all_samples})
summary["systems_val"] = summary["systems_train"]
with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps({
    "full_data_no_split": True,
    "samples": len(all_samples),
    "records": summary["train_records"],
    "original_split_samples": summary["original_split_samples"],
}, ensure_ascii=False))
PY
}

write_model_config() {
  local property_name="$1"
  local seed="$2"
  local experiment_time="$3"
  local destination="$4"
  local training_pct="$5"
  local use_phy="$6"
  local use_rk="$7"
  local rk_order="$8"
  local source_config
  source_config="$(resolve_config_path "$property_name")" || {
    echo "unsupported property: $property_name" >&2
    exit 1
  }
  python - "$source_config" "$destination" "$seed" "$experiment_time" "$property_name" "$training_pct" "$use_phy" "$use_rk" "$rk_order" <<'PY'
import sys
from pathlib import Path
import yaml

source_path = Path(sys.argv[1])
destination_path = Path(sys.argv[2])
seed = int(sys.argv[3])
experiment_time = sys.argv[4]
property_name = sys.argv[5]
training_pct = int(sys.argv[6])
use_phy = sys.argv[7].strip().lower() == "true"
use_rk = sys.argv[8].strip().lower() == "true"
raw_rk_order = sys.argv[9].strip()
rk_order = None if raw_rk_order.lower() == "none" else int(raw_rk_order)

config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
config["training"]["seed"] = seed
config["training"]["low_data_learning"] = False if training_pct == 100 else f"true,{training_pct / 100:.2f}"
config["model"]["use_phy"] = use_phy
config["model"]["use_arrhenius"] = False
config["model"]["use_linear"] = False
config["model"]["use_rk"] = use_rk
if rk_order is None:
    config["model"]["rk_order"] = 0
else:
    config["model"]["rk_order"] = rk_order
raw_dir_key = {
    "density": "raw_density_dir",
    "viscosity": "raw_viscosity_dir",
    "surface_tension": "raw_surface_tension_dir",
    "electrical_conductivity": "raw_electrical_conductivity_dir",
}[property_name]
config["data"][raw_dir_key] = f"/root/shared-nvme/Salt/data/{property_name}"
n_label = "none" if rk_order is None else str(rk_order)
model_setting = f"training_{training_pct}pct_phys_{str(use_phy).lower()}_rk_{str(use_rk).lower()}_n_{n_label}"
config["training"]["output_dir"] = f"/root/shared-nvme/Salt/outputs/{experiment_time}/{property_name}/{model_setting}/seed={seed}"
destination_path.parent.mkdir(parents=True, exist_ok=True)
destination_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

write_formula_space_config() {
  local property_name="$1"
  local seed="$2"
  local experiment_time="$3"
  local destination="$4"
  local use_phy="$5"
  local use_rk="$6"
  local use_arrhenius="$7"
  local rk_order="$8"
  local batch_size="${FORMULA_SPACE_BATCH_SIZE:-128}"
  local source_config
  source_config="$(resolve_config_path "$property_name")" || {
    echo "unsupported property: $property_name" >&2
    exit 1
  }
  python - "$source_config" "$destination" "$seed" "$experiment_time" "$property_name" "$use_phy" "$use_rk" "$use_arrhenius" "$rk_order" "$batch_size" <<'PY'
import sys
from pathlib import Path
import yaml

source_path = Path(sys.argv[1])
destination_path = Path(sys.argv[2])
seed = int(sys.argv[3])
experiment_time = sys.argv[4]
property_name = sys.argv[5]
use_phy = sys.argv[6].strip().lower() == "true"
use_rk = sys.argv[7].strip().lower() == "true"
use_arrhenius = sys.argv[8].strip().lower() == "true"
raw_rk_order = sys.argv[9].strip()
batch_size = int(sys.argv[10])
rk_order = None if raw_rk_order.lower() == "none" else int(raw_rk_order)

config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
config["training"]["seed"] = seed
config["training"]["batch_size"] = batch_size
config["training"]["low_data_learning"] = False
config["model"]["use_phy"] = use_phy
config["model"]["use_rk"] = use_rk
config["model"]["use_arrhenius"] = use_arrhenius
config["model"]["use_linear"] = False
config["model"]["rk_order"] = 0 if rk_order is None else rk_order
raw_dir_key = {
    "density": "raw_density_dir",
    "viscosity": "raw_viscosity_dir",
    "surface_tension": "raw_surface_tension_dir",
    "electrical_conductivity": "raw_electrical_conductivity_dir",
}[property_name]
config["data"][raw_dir_key] = f"/root/shared-nvme/Salt/data/{property_name}"
n_label = "none" if rk_order is None else str(rk_order)
if not use_phy:
    model_setting = "phys_false"
elif use_arrhenius:
    model_setting = "phys_true_arrhenius"
elif not use_rk:
    model_setting = "phys_true_rk_false"
else:
    model_setting = f"phys_true_rk_true_n_{n_label}"
config["training"]["output_dir"] = (
    f"/root/shared-nvme/Salt/outputs/{experiment_time}/{property_name}/{model_setting}/seed={seed}"
)
destination_path.parent.mkdir(parents=True, exist_ok=True)
destination_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

write_linear_formula_config() {
  local property_name="$1"
  local seed="$2"
  local experiment_time="$3"
  local destination="$4"
  local batch_size="${FORMULA_SPACE_BATCH_SIZE:-128}"
  local source_config
  source_config="$(resolve_config_path "$property_name")" || {
    echo "unsupported property: $property_name" >&2
    exit 1
  }
  python - "$source_config" "$destination" "$seed" "$experiment_time" "$property_name" "$batch_size" <<'PY'
import sys
from pathlib import Path
import yaml

source_path = Path(sys.argv[1])
destination_path = Path(sys.argv[2])
seed = int(sys.argv[3])
experiment_time = sys.argv[4]
property_name = sys.argv[5]
batch_size = int(sys.argv[6])

config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
config["training"]["seed"] = seed
config["training"]["batch_size"] = batch_size
config["training"]["low_data_learning"] = False
config["model"]["use_phy"] = True
config["model"]["use_rk"] = False
config["model"]["use_arrhenius"] = False
config["model"]["use_linear"] = True
config["model"]["rk_order"] = 0
raw_dir_key = {
    "density": "raw_density_dir",
    "surface_tension": "raw_surface_tension_dir",
}[property_name]
config["data"][raw_dir_key] = f"/root/shared-nvme/Salt/data/{property_name}"
model_setting = "phys_true_linear"
config["training"]["output_dir"] = (
    f"/root/shared-nvme/Salt/outputs/{experiment_time}/{property_name}/{model_setting}/seed={seed}"
)
destination_path.parent.mkdir(parents=True, exist_ok=True)
destination_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

write_seed42_n4_interval_config() {
  local property_name="$1"
  local seed="$2"
  local experiment_time="$3"
  local destination="$4"
  local source_config
  source_config="$(resolve_config_path "$property_name")" || {
    echo "unsupported property: $property_name" >&2
    exit 1
  }
  python - "$source_config" "$destination" "$seed" "$experiment_time" "$property_name" <<'PY'
import sys
from pathlib import Path
import yaml

source_path = Path(sys.argv[1])
destination_path = Path(sys.argv[2])
seed = int(sys.argv[3])
experiment_time = sys.argv[4]
property_name = sys.argv[5]

config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
config["data"]["validation_ratio"] = 0.2
config["data"]["temperature_interval"] = 50
config["data"]["composition_interval"] = 5
config["training"]["seed"] = seed
config["training"]["low_data_learning"] = False
config["training"]["batch_size"] = 128
config["model"]["use_phy"] = True
config["model"]["use_arrhenius"] = False
config["model"]["use_linear"] = False
config["model"]["use_rk"] = True
config["model"]["rk_order"] = 4
raw_dir_key = {
    "density": "raw_density_dir",
    "viscosity": "raw_viscosity_dir",
    "surface_tension": "raw_surface_tension_dir",
    "electrical_conductivity": "raw_electrical_conductivity_dir",
}[property_name]
config["data"][raw_dir_key] = f"/root/shared-nvme/Salt/data/{property_name}"
model_setting = "phys_true_rk_true_n_4"
config["training"]["output_dir"] = f"/root/shared-nvme/Salt/outputs/{experiment_time}/{property_name}/{model_setting}/seed={seed}"
destination_path.parent.mkdir(parents=True, exist_ok=True)
destination_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

if [[ -n "$ARCHIVE_NAME" ]]; then
  tar -xzf "/tmp/$ARCHIVE_NAME"
fi

python3 -m venv --system-site-packages .venv || true
source .venv/bin/activate

python -c 'import torch; print({"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()})'
python -c 'import importlib, subprocess, sys; required={"numpy":"numpy","pandas":"pandas","yaml":"PyYAML"}; missing=[pkg for mod,pkg in required.items() if importlib.util.find_spec(mod) is None]; print({"dependency_check": missing or "all_present"}); subprocess.check_call([sys.executable, "-m", "pip", "install", *missing]) if missing else None'

run_property() {
  local property_name="$1"
  local cli_property
  cli_property="$(cli_property_name "$property_name")"
  local config_path
  config_path="$(resolve_config_path "$property_name")" || {
    echo "unsupported property: $property_name" >&2
    exit 1
  }
  local work_dir
  work_dir="$(python - "$config_path" <<'PY'
import sys, yaml
from pathlib import Path
config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["training"]["output_dir"])
PY
)"
  run_with_config "$cli_property" "$config_path" "$work_dir"
}

if [[ "$PROPERTY" == "all" ]]; then
  properties=(density viscosity surface_tension electrical_conductivity)
  pids=()
  for property_name in "${properties[@]}"; do
    (
      run_property "$property_name"
    ) >"$RUN_DIR/${property_name}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
elif [[ "$PROPERTY" == "all_n_serial" ]]; then
  properties=(density viscosity surface_tension electrical_conductivity)
  rk_orders=(None 0 2 4)
  seeds=(42 43 44 45 46)
  for property_name in "${properties[@]}"; do
    for rk_order in "${rk_orders[@]}"; do
      if [[ "$rk_order" == "None" ]]; then
        rk_status="false"
      else
        rk_status="true"
      fi
      model_setting="$(model_setting_slug 100 true "$rk_status" "$rk_order")"
      n_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}"
      mkdir -p "$n_dir"
      for seed in "${seeds[@]}"; do
        work_dir="$n_dir/seed=${seed}"
        mkdir -p "$work_dir"
        config_path="$work_dir/config.yaml"
        write_model_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path" 100 true "$rk_status" "$rk_order"
        printf 'starting %s RK=%s N=%s seed=%s\n' "$property_name" "$rk_status" "$rk_order" "$seed" >"$work_dir/run.log"
        (
          cd "$work_dir"
          PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
          PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
          PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
        ) >>"$work_dir/run.log" 2>&1
      done
    done
  done
elif [[ "$PROPERTY" == "all_non_phy_serial" ]]; then
  properties=(density viscosity surface_tension electrical_conductivity)
  seeds=(42 43 44 45 46)
  for property_name in "${properties[@]}"; do
    model_setting="$(model_setting_slug 100 false false None)"
    model_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}"
    for seed in "${seeds[@]}"; do
      work_dir="${model_dir}/seed=${seed}"
      mkdir -p "$work_dir"
      config_path="$work_dir/config.yaml"
      write_model_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path" 100 false false None
      printf 'starting %s Non_phy seed=%s\n' "$property_name" "$seed" >"$work_dir/run.log"
      (
        cd "$work_dir"
        PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
        PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
        PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
      ) >>"$work_dir/run.log" 2>&1
    done
  done
elif [[ "$PROPERTY" == "all_low_data_serial" ]]; then
  run_fractional_serial 20 \
    "false false None"
    "true false None"
    "true true 0"
    "true true 2"
    "true true 4"
elif [[ "$PROPERTY" == "all_low_data_rk24_serial" ]]; then
  run_fractional_serial 20 \
    "true true 2"
    "true true 4"
elif [[ "$PROPERTY" == "all_fractional_serial" ]]; then
  run_fractional_serial "$TRAINING_PCT" \
    "false false None" \
    "true false None" \
    "true true 0" \
    "true true 2" \
    "true true 4"
elif [[ "$PROPERTY" == "all_seed42_n4_interval_serial" ]]; then
  properties=(density viscosity surface_tension electrical_conductivity)
  seed=42
  model_setting="phys_true_rk_true_n_4"
  for property_name in "${properties[@]}"; do
    work_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}/seed=${seed}"
    mkdir -p "$work_dir"
    config_path="$work_dir/config.yaml"
    write_seed42_n4_interval_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path"
    printf 'starting %s phys=true rk=true n=4 seed=%s batch_size=128 record_split=80/20 temperature_interval=50K composition_interval=5%%\n' "$property_name" "$seed" >"$work_dir/run.log"
    (
      cd "$work_dir"
      PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
      PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
      PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
    ) >>"$work_dir/run.log" 2>&1
  done
elif [[ "$PROPERTY" == "all_seed42_formula_space" ]]; then
  properties=(viscosity electrical_conductivity density surface_tension)
  if [[ -n "${FORMULA_SPACE_SEEDS:-}" ]]; then
    read -r -a seeds <<<"$FORMULA_SPACE_SEEDS"
  else
    seeds=("${FORMULA_SPACE_SEED:-42}")
  fi
  batch_size="${FORMULA_SPACE_BATCH_SIZE:-128}"
  task_properties=()
  task_specs=()
  task_seeds=()
  common_specs=(
    "false false false None"
    "true false false None"
    "true true false 0"
    "true true false 2"
    "true true false 4"
  )
  for property_name in "${properties[@]}"; do
    specs=("${common_specs[@]}")
    if [[ "$property_name" == "viscosity" || "$property_name" == "electrical_conductivity" ]]; then
      specs+=("true false true None")
    fi
    for spec in "${specs[@]}"; do
      read -r use_phy use_rk use_arrhenius rk_order <<<"$spec"
      model_setting="$(formula_model_setting_slug "$use_phy" "$use_rk" "$use_arrhenius" "$rk_order")"
      for seed in "${seeds[@]}"; do
        task_properties+=("$property_name")
        task_specs+=("$spec")
        task_seeds+=("$seed")
        work_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}/seed=${seed}"
        mkdir -p "$work_dir"
        config_path="$work_dir/config.yaml"
        write_formula_space_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path" "$use_phy" "$use_rk" "$use_arrhenius" "$rk_order"
        printf 'starting %s model_setting=%s phys=%s rk=%s arrhenius=%s n=%s seed=%s batch_size=%s full_data=true\n' \
          "$property_name" "$model_setting" "$use_phy" "$use_rk" "$use_arrhenius" "$(normalize_n_label "$rk_order")" "$seed" "$batch_size" >"$work_dir/run.log"
      done
    done
  done
  for index in "${!task_properties[@]}"; do
    property_name="${task_properties[$index]}"
    read -r use_phy use_rk use_arrhenius rk_order <<<"${task_specs[$index]}"
    seed="${task_seeds[$index]}"
    model_setting="$(formula_model_setting_slug "$use_phy" "$use_rk" "$use_arrhenius" "$rk_order")"
    work_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}/seed=${seed}"
    config_path="$work_dir/config.yaml"
    (
      cd "$work_dir"
      PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
      PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
      PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
      ) >>"$work_dir/run.log" 2>&1
  done
elif [[ "$PROPERTY" == "selected_full_data" ]]; then
  properties=(density surface_tension viscosity electrical_conductivity)
  seeds=(42 43 44 45 46)
  batch_size="${FULL_DATA_BATCH_SIZE:-64}"
  for property_name in "${properties[@]}"; do
    for seed in "${seeds[@]}"; do
      case "$property_name" in
        density)
          model_setting="phys_true_rk_true_n_0"
          work_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}/seed=${seed}"
          mkdir -p "$work_dir"
          config_path="$work_dir/config.yaml"
          write_model_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path" 100 true true 0
          ;;
        surface_tension)
          model_setting="$(linear_model_setting_slug true true)"
          work_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}/seed=${seed}"
          mkdir -p "$work_dir"
          config_path="$work_dir/config.yaml"
          FORMULA_SPACE_BATCH_SIZE="$batch_size" write_linear_formula_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path"
          ;;
        viscosity|electrical_conductivity)
          model_setting="phys_true_arrhenius"
          work_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}/seed=${seed}"
          mkdir -p "$work_dir"
          config_path="$work_dir/config.yaml"
          FORMULA_SPACE_BATCH_SIZE="$batch_size" write_formula_space_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path" true false true None
          ;;
        *)
          echo "unsupported selected_full_data property: $property_name" >&2
          exit 1
          ;;
      esac
      python - "$config_path" "$batch_size" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
batch_size = int(sys.argv[2])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
config["training"]["batch_size"] = batch_size
config["training"]["low_data_learning"] = False
config["training"]["output_dir"] = str(config_path.parent)
config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
      printf 'starting %s model_setting=%s seed=%s batch_size=%s full_data_no_split=true\n' \
        "$property_name" "$model_setting" "$seed" "$batch_size" >"$work_dir/run.log"
      (
        cd "$work_dir"
        PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
        use_full_data_no_split "$work_dir"
        PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
        PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
      ) >>"$work_dir/run.log" 2>&1
    done
  done
elif [[ "$PROPERTY" == "density_surface_linear" ]]; then
  properties=(density surface_tension)
  if [[ -n "${FORMULA_SPACE_SEEDS:-}" ]]; then
    read -r -a seeds <<<"$FORMULA_SPACE_SEEDS"
  else
    seeds=(42 43 44 45 46)
  fi
  batch_size="${FORMULA_SPACE_BATCH_SIZE:-64}"
  model_setting="$(linear_model_setting_slug true true)"
  for property_name in "${properties[@]}"; do
    for seed in "${seeds[@]}"; do
      work_dir="/root/shared-nvme/Salt/outputs/${EXPERIMENT_TIME}/${property_name}/${model_setting}/seed=${seed}"
      if [[ -f "$work_dir/metrics.json" ]]; then
        echo "skipping completed ${property_name}/${model_setting}/seed=${seed}"
        continue
      fi
      mkdir -p "$work_dir"
      config_path="$work_dir/config.yaml"
      FORMULA_SPACE_BATCH_SIZE="$batch_size" write_linear_formula_config "$property_name" "$seed" "$EXPERIMENT_TIME" "$config_path"
      printf 'starting %s model_setting=%s phys=true linear=true rk=false arrhenius=false seed=%s batch_size=%s full_data=true\n' \
        "$property_name" "$model_setting" "$seed" "$batch_size" >"$work_dir/run.log"
      (
        cd "$work_dir"
        PYTHONPATH="$REPO_ROOT" python -m salt_property preprocess --config "$config_path" --property "$property_name"
        PYTHONPATH="$REPO_ROOT" python -m salt_property train --config "$config_path" --property "$property_name"
        PYTHONPATH="$REPO_ROOT" python -m salt_property eval --config "$config_path" --property "$property_name"
      ) >>"$work_dir/run.log" 2>&1
    done
  done
else
  run_property "$PROPERTY"
fi
