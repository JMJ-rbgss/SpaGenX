#!/usr/bin/env bash
# PyTorch-based frozen WSI pipeline. Preview by default; --execute runs it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:---dry-run}"
if [[ "$MODE" != --dry-run && "$MODE" != --execute ]]; then
  echo 'Usage: bash scripts/infer_wsi.sh [--dry-run|--execute]' >&2; exit 2
fi
: "${WSI:?Set WSI to an existing image path}"
: "${ORGAN:?Set ORGAN to a frozen codebook organ name}"
: "${GENES_FILE:?Set GENES_FILE to a one-gene-per-line file}"
: "${RUN_DIR:?Set RUN_DIR to a new absolute output directory}"
[[ "$RUN_DIR" = /* ]] || { echo 'RUN_DIR must be absolute' >&2; exit 2; }
[[ -f "$WSI" && -f "$GENES_FILE" ]] || { echo 'Missing image or gene list' >&2; exit 2; }
[[ -f "$ROOT/bundle_config.json" ]] || { echo 'Configure bundle_config.json first' >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo 'Output already exists; use a fresh RUN_DIR' >&2; exit 2; }
DEVICE="${DEVICE:-cuda:0}"
pre=("$PYTHON_BIN" "$ROOT/inference/spagenx_cli.py" --bundle "$ROOT" preprocess --image "$WSI" --out "$RUN_DIR/preprocessed" --device "$DEVICE")
if [[ -n "${MPP:-}" ]]; then pre+=(--mpp "$MPP"); fi
inf=("$PYTHON_BIN" "$ROOT/inference/spagenx_cli.py" --bundle "$ROOT" infer --features "$RUN_DIR/preprocessed/features.pt" --neighbors "$RUN_DIR/preprocessed/neighbors.pt" --genes-file "$GENES_FILE" --organ "$ORGAN" --out "$RUN_DIR/predictions" --device "$DEVICE" --patch-batch "${PATCH_BATCH:-16}" --gene-chunk "${GENE_CHUNK:-256}")
printf 'PREPROCESS:'; printf ' %q' "${pre[@]}"; printf '\n'
printf 'INFERENCE:'; printf ' %q' "${inf[@]}"; printf '\n'
if [[ "$MODE" == --dry-run ]]; then echo 'DRY_RUN: no image processed and no model loaded'; exit 0; fi
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
"${pre[@]}" >"$RUN_DIR/preprocess.log" 2>&1
"${inf[@]}" >"$RUN_DIR/inference.log" 2>&1
echo "Completed: $RUN_DIR/predictions/predictions.npz"
