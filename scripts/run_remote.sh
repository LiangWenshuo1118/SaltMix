#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${REMOTE_PORT:-2222}"
REMOTE_USER="${REMOTE_USER:-root@ackcs-00gjgrdb}"
REMOTE_HOST="${REMOTE_HOST:-ssh.zw1.paratera.com}"
REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
PASSWORD="${REMOTE_PASSWORD:-288eb8e9921861f3f9fcf84e43eee0c0}"
PROPERTY="${1:-viscosity}"
STAMP="${2:-$(date +%Y_%m_%d_%H_%M)}"
MODE="${3:-foreground}"
TRAINING_PCT="${TRAINING_PCT:-20}"

case "$PROPERTY" in
  all)
    REMOTE_BASE="/root/shared-nvme/Salt/all_property_runs"
    ;;
  all_n_serial|all_non_phy_serial|all_low_data_serial|all_low_data_rk24_serial|all_fractional_serial|all_seed42_n4_interval_serial|all_seed42_formula_space|selected_full_data)
    REMOTE_BASE="/root/shared-nvme/Salt"
    ;;
  density)
    REMOTE_BASE="/root/shared-nvme/Salt/density_rk_runs"
    ;;
  viscosity)
    REMOTE_BASE="/root/shared-nvme/Salt/viscosity_rk_runs"
    ;;
  viscosity_arrhenius)
    REMOTE_BASE="/root/shared-nvme/Salt/viscosity_arrhenius_runs"
    ;;
  surface_tension)
    REMOTE_BASE="/root/shared-nvme/Salt/surface_tension_rk_runs"
    ;;
  electrical_conductivity)
    REMOTE_BASE="/root/shared-nvme/Salt/electrical_conductivity_rk_runs"
    ;;
  electrical_conductivity_arrhenius)
    REMOTE_BASE="/root/shared-nvme/Salt/electrical_conductivity_arrhenius_runs"
    ;;
  *)
    echo "unsupported property: $PROPERTY" >&2
    exit 1
    ;;
esac

ARCHIVE="$("$ROOT_DIR/scripts/package_for_remote.sh" "$PROPERTY" "$STAMP")"
ARCHIVE_NAME="$(basename "$ARCHIVE")"
if [[ "$PROPERTY" == "all_n_serial" || "$PROPERTY" == "all_non_phy_serial" || "$PROPERTY" == "all_low_data_serial" || "$PROPERTY" == "all_low_data_rk24_serial" || "$PROPERTY" == "all_fractional_serial" || "$PROPERTY" == "all_seed42_n4_interval_serial" || "$PROPERTY" == "all_seed42_formula_space" || "$PROPERTY" == "selected_full_data" ]]; then
  REMOTE_RUN_DIR="$REMOTE_BASE"
else
  REMOTE_RUN_DIR="$REMOTE_BASE/$STAMP"
fi
REMOTE_ENV_PREFIX=""
if [[ "$PROPERTY" == "all_fractional_serial" ]]; then
  REMOTE_ENV_PREFIX="TRAINING_PCT=$TRAINING_PCT "
fi
REMOTE_COMMAND="set -euo pipefail; mkdir -p $REMOTE_RUN_DIR; cd $REMOTE_RUN_DIR; tar --no-same-owner --no-same-permissions -xzf /tmp/$ARCHIVE_NAME; ${REMOTE_ENV_PREFIX}bash scripts/remote_train_entry.sh $PROPERTY $REMOTE_RUN_DIR '' $STAMP"
if [[ "$PROPERTY" == "all_seed42_formula_space" && -n "${FORMULA_SPACE_SEED:-}" ]]; then
  REMOTE_ENV_PREFIX="FORMULA_SPACE_SEED=$FORMULA_SPACE_SEED "
  if [[ -n "${FORMULA_SPACE_BATCH_SIZE:-}" ]]; then
    REMOTE_ENV_PREFIX="${REMOTE_ENV_PREFIX}FORMULA_SPACE_BATCH_SIZE=$FORMULA_SPACE_BATCH_SIZE "
  fi
  REMOTE_COMMAND="set -euo pipefail; mkdir -p $REMOTE_RUN_DIR; cd $REMOTE_RUN_DIR; tar --no-same-owner --no-same-permissions -xzf /tmp/$ARCHIVE_NAME; ${REMOTE_ENV_PREFIX}bash scripts/remote_train_entry.sh $PROPERTY $REMOTE_RUN_DIR '' $STAMP"
fi
if [[ "$PROPERTY" == "all_seed42_formula_space" && -n "${FORMULA_SPACE_SEEDS:-}" ]]; then
  REMOTE_ENV_PREFIX="FORMULA_SPACE_SEEDS='$FORMULA_SPACE_SEEDS' "
  if [[ -n "${FORMULA_SPACE_BATCH_SIZE:-}" ]]; then
    REMOTE_ENV_PREFIX="${REMOTE_ENV_PREFIX}FORMULA_SPACE_BATCH_SIZE=$FORMULA_SPACE_BATCH_SIZE "
  fi
  REMOTE_COMMAND="set -euo pipefail; mkdir -p $REMOTE_RUN_DIR; cd $REMOTE_RUN_DIR; tar --no-same-owner --no-same-permissions -xzf /tmp/$ARCHIVE_NAME; ${REMOTE_ENV_PREFIX}bash scripts/remote_train_entry.sh $PROPERTY $REMOTE_RUN_DIR '' $STAMP"
fi
if [[ "$MODE" == "detach" ]]; then
  REMOTE_LOG="$REMOTE_RUN_DIR/${PROPERTY}_${STAMP}.log"
  if [[ "$PROPERTY" == "all_seed42_formula_space" && -n "${FORMULA_SPACE_SEED:-}" ]]; then
    REMOTE_LOG="$REMOTE_RUN_DIR/${PROPERTY}_${STAMP}_seed${FORMULA_SPACE_SEED}.log"
    if [[ -n "${FORMULA_SPACE_BATCH_SIZE:-}" ]]; then
      REMOTE_LOG="$REMOTE_RUN_DIR/${PROPERTY}_${STAMP}_seed${FORMULA_SPACE_SEED}_batch${FORMULA_SPACE_BATCH_SIZE}.log"
    fi
  fi
  if [[ "$PROPERTY" == "all_seed42_formula_space" && -n "${FORMULA_SPACE_SEEDS:-}" ]]; then
    REMOTE_LOG="$REMOTE_RUN_DIR/${PROPERTY}_${STAMP}_seeds$(printf '%s' "$FORMULA_SPACE_SEEDS" | tr ' ' '_').log"
    if [[ -n "${FORMULA_SPACE_BATCH_SIZE:-}" ]]; then
      REMOTE_LOG="$REMOTE_RUN_DIR/${PROPERTY}_${STAMP}_seeds$(printf '%s' "$FORMULA_SPACE_SEEDS" | tr ' ' '_')_batch${FORMULA_SPACE_BATCH_SIZE}.log"
    fi
  fi
  REMOTE_COMMAND="set -euo pipefail; mkdir -p $REMOTE_RUN_DIR; cd $REMOTE_RUN_DIR; tar --no-same-owner --no-same-permissions -xzf /tmp/$ARCHIVE_NAME; ${REMOTE_ENV_PREFIX}nohup bash scripts/remote_train_entry.sh $PROPERTY $REMOTE_RUN_DIR '' $STAMP > $REMOTE_LOG 2>&1 < /dev/null & printf 'remote_log=%s\n' $REMOTE_LOG"
elif [[ "$MODE" != "foreground" ]]; then
  echo "unsupported mode: $MODE" >&2
  exit 1
fi

/usr/bin/expect <<EOF
set timeout -1
spawn scp -O -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa -P $PORT "$ARCHIVE" "$REMOTE_TARGET:/tmp/$ARCHIVE_NAME"
expect "password:" { send "$PASSWORD\r" }
expect eof
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa -p $PORT "$REMOTE_TARGET" "bash -lc \"$REMOTE_COMMAND\""
expect "password:" { send "$PASSWORD\r" }
expect eof
EOF

printf '%s\n' "$REMOTE_RUN_DIR"
