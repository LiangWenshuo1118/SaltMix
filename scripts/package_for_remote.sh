#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROPERTY="${1:-viscosity}"
STAMP="${2:-$(date +%Y_%m_%d_%H_%M)}"
BUNDLE_DIR="$ROOT_DIR/outputs/remote_bundle"
mkdir -p "$BUNDLE_DIR"

case "$PROPERTY" in
  all)
    ARCHIVE_PREFIX="salt_property_all"
    ;;
  all_n_serial)
    ARCHIVE_PREFIX="salt_property_all_n_serial"
    ;;
  all_non_phy_serial)
    ARCHIVE_PREFIX="salt_property_all_non_phy_serial"
    ;;
  all_low_data_serial)
    ARCHIVE_PREFIX="salt_property_all_low_data_serial"
    ;;
  all_low_data_rk24_serial)
    ARCHIVE_PREFIX="salt_property_all_low_data_rk24_serial"
    ;;
  all_fractional_serial)
    ARCHIVE_PREFIX="salt_property_all_fractional_serial"
    ;;
  all_seed42_n4_interval_serial)
    ARCHIVE_PREFIX="salt_property_seed42_n4_interval"
    ;;
  all_seed42_formula_space)
    ARCHIVE_PREFIX="salt_property_seed42_formula_space"
    ;;
  selected_full_data)
    ARCHIVE_PREFIX="salt_property_selected_full_data"
    ;;
  density_surface_linear)
    ARCHIVE_PREFIX="salt_property_linear_density_surface_seed42_46"
    ;;
  density)
    ARCHIVE_PREFIX="density_rk"
    ;;
  viscosity)
    ARCHIVE_PREFIX="viscosity_rk"
    ;;
  viscosity_arrhenius)
    ARCHIVE_PREFIX="viscosity_arrhenius"
    ;;
  surface_tension)
    ARCHIVE_PREFIX="surface_tension_rk"
    ;;
  electrical_conductivity)
    ARCHIVE_PREFIX="electrical_conductivity_rk"
    ;;
  electrical_conductivity_arrhenius)
    ARCHIVE_PREFIX="electrical_conductivity_arrhenius"
    ;;
  *)
    echo "unsupported property: $PROPERTY" >&2
    exit 1
    ;;
esac

ARCHIVE="$BUNDLE_DIR/${ARCHIVE_PREFIX}_${STAMP}.tar.gz"
COPYFILE_DISABLE=1 tar \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='outputs/remote_bundle' \
  --exclude='outputs/density_results' \
  --exclude='outputs/density_linear_results' \
  --exclude='outputs/viscosity_results' \
  --exclude='outputs/viscosity_arrhenius_results' \
  --exclude='outputs/surface_tension_results' \
  --exclude='outputs/surface_tension_linear_results' \
  --exclude='outputs/electrical_conductivity_results' \
  --exclude='outputs/electrical_conductivity_arrhenius_results' \
  -czf "$ARCHIVE" \
  -C "$ROOT_DIR" \
  salt_property \
  configs \
  scripts \
  requirements-remote.txt \
  data/density \
  data/viscosity \
  data/surface_tension \
  data/electrical_conductivity

printf '%s\n' "$ARCHIVE"
