#!/usr/bin/env bash
# =============================================================================
# run_local.sh -- run the SVD-LLM baseline on a single local GPU box. No Slurm.
#
# This is the Slurm-free equivalent of the Trillium runner (baselines/run.sh in
# the ARKS comparison harness). Same compression, same evaluation, same output
# JSON -- the only thing removed is the batch scheduler.
#
# WHAT THIS MEASURES
#   SVD-LLM performs the compression (whitening + truncated SVD). Perplexity
#   comes from local/shared_eval.py, which transcribes the ARKS evaluation
#   recipe, NOT from evaluater.py -- so the number is comparable to the other
#   baselines and to ARKS.
#
# RATIO CONVENTION -- read this before changing anything
#   SVDLLM.py's --ratio help text claims it is the fraction KEPT, but
#   SVDLLM.py does `args.ratio = 1 - args.ratio` before use, so the CLI value is
#   really the fraction REMOVED. local/run_svdllm.py takes --rho = fraction KEPT
#   and passes it straight to whitening(), which is the already-inverted value
#   the internals expect. There is no double inversion. Do not "fix" this.
#
# SINGLE-GPU / LAYERWISE -- the honest answer for this baseline
#   Of the three SVD baselines this is the one that is already almost entirely
#   single-GPU and layerwise.
#     * Profiling: profle_svdllm_low_resource (SVDLLM.py) is upstream's own
#       layer-streaming implementation. The model stays on the HOST; only the
#       embeddings/norms plus ONE decoder layer are on the GPU at a time. This
#       is the default here for every model except opt-125m.
#       profle_svdllm, the alternative, does `model.to(dev)` -- whole model.
#     * Whitening + truncated SVD: per-MATRIX, finer than layerwise -- one
#       matrix's d_in x d_in Cholesky work on the device at a time. Size the
#       card against FOUR such fp64 buffers, not one: the input and the
#       out-of-place factor are both live, and the escalating-ridge retry adds
#       two more. On the largest ffn_dim that is ~6.6 GB for Llama-3.1-8B,
#       ~13.4 GB for opt-13b, ~26.3 GB for opt-30b. The retry path is the
#       expected one at 6.7B and above, not the exception.
#     * The Cholesky factors for ALL layers would otherwise stay resident while
#       whitening consumes them one layer at a time (176 GB on opt-13b, 414 GB
#       on opt-30b of HOST RAM). SVDLLM_SPILL_DIR pages them to disk; it is
#       byte-exact and changes no number, only where the bytes live.
#     * Evaluation: whole-model by default. SHARED_EVAL_LAYERWISE=1 streams
#       decoder layers instead -- needed for models that do not fit one card.
#   NO BACKWARD PASS runs on this path. whitening_local_update / local_update /
#   fasterprune are all @torch.no_grad() closed-form updates, not training. The
#   only real backward in the repo is utils/LoRA.py (HF Trainer, whole model on
#   GPU), which this runner never invokes.
#
# USAGE
#   ./run_local.sh setup [--force]        build the venv (transformers 4.45.2)
#   ./run_local.sh doctor                 check GPU, venv, imports
#   ./run_local.sh run MODEL RHO          one (model, rho)
#   ./run_local.sh sweep [MODEL ...]      every model x every rho, serially
#   ./run_local.sh collect                runs/*.json -> results.csv
#
#   ./run_local.sh run facebook/opt-1.3b 0.60
#   LAYERWISE_EVAL=1 ./run_local.sh run facebook/opt-66b 0.60
#
# ENVIRONMENT KNOBS (all optional)
#   VENV=<dir>        venv location                  default ./.venv-svdllm
#   PYBIN=<python>    interpreter used to build it   default python3.11 else python3
#   RUNS_DIR=<dir>    result JSONs                   default ./local/runs/svdllm
#   SPILL_ROOT=<dir>  where Cholesky factors spill   default ./local/.spill
#                     Needs up to ~414 GB free for opt-30b. Put it on fast local
#                     NVMe, not a network filesystem.
#   LOG_DIR, RHOS, MODELS, SEED, GPUS, OFFLINE, TORCH_SPEC, EXTRA_ARGS
#                     -- as in the other two baselines' run_local.sh
#   LOW_RESOURCE=auto|1|0   layer-streamed profiling. auto (default) = on for
#                     everything except opt-125m, matching the reference runs.
#   SPILL=auto|1|0    auto (default) = on for 13b/30b/66b/70B, matching the
#                     reference runs. HOST_DTYPE=source is added for >13B.
#   LAYERWISE_EVAL=1  set SHARED_EVAL_LAYERWISE=1 for the perplexity pass.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

METHOD=svdllm
VENV="${VENV:-$HERE/.venv-svdllm}"
RUNS_DIR="${RUNS_DIR:-$HERE/local/runs/$METHOD}"
LOG_DIR="${LOG_DIR:-$HERE/local/logs}"
SPILL_ROOT="${SPILL_ROOT:-$HERE/local/.spill}"
RHOS="${RHOS:-0.60 0.33}"
SEED="${SEED:-42}"
GPUS="${GPUS:-0}"

# Pinned to exactly what the reference runs resolved to. Both exist on PyPI.
# transformers 4.45.2 is NOT negotiable: SVDOPTDecoderLayer.forward() does not
# accept `position_ids`, which 4.48+'s OPTDecoder passes unconditionally.
TORCH_SPEC="${TORCH_SPEC:-torch==2.14.0}"
TRANSFORMERS_SPEC="transformers==4.45.2"
NUMPY_SPEC="numpy==1.26.4"

MODELS="${MODELS:-facebook/opt-125m facebook/opt-350m facebook/opt-1.3b \
facebook/opt-2.7b meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B \
facebook/opt-6.7b facebook/opt-13b facebook/opt-30b meta-llama/Llama-3.1-8B}"

log() { echo "[run_local] $*"; }
die() { echo "[run_local] ERROR: $*" >&2; exit 1; }
rho_tag() { awk -v r="$1" 'BEGIN{printf "%.0f", r*100}'; }

setup_env() {
  mkdir -p "$RUNS_DIR" "$LOG_DIR"
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  [[ "$GPUS" == "all" ]] || export CUDA_VISIBLE_DEVICES="$GPUS"
  local t="${LOCAL_CPUS:-$(nproc)}"
  export OMP_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" MKL_NUM_THREADS="$t"
  [[ "${OFFLINE:-0}" == "1" ]] && export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
  [[ "${LAYERWISE_EVAL:-0}" == "1" ]] && export SHARED_EVAL_LAYERWISE=1
  return 0
}

do_setup() {
  local force=0; [[ "${1:-}" == "--force" ]] && force=1
  local py="${PYBIN:-}"
  if [[ -z "$py" ]]; then
    py=$(command -v python3.11 || command -v python3) || die "no python3 on PATH"
  fi
  if [[ -d "$VENV" && $force -eq 0 ]]; then
    log "$VENV exists -- reusing it (./run_local.sh setup --force to rebuild)"
  else
    (( force )) && rm -rf "$VENV"
    log "building $VENV with $py ($("$py" -V 2>&1))"
    "$py" -m venv "$VENV"
  fi
  "$VENV/bin/pip" install --upgrade pip setuptools wheel
  "$VENV/bin/pip" install "$TORCH_SPEC" "$NUMPY_SPEC" "$TRANSFORMERS_SPEC" \
      scipy scikit_learn safetensors sentencepiece datasets accelerate \
      pyarrow \
      pandas tqdm pyyaml protobuf
  "$VENV/bin/python" - <<'PY'
import torch, transformers, numpy
print(f"  transformers {transformers.__version__}  torch {torch.__version__} "
      f"(cuda {torch.version.cuda})  numpy {numpy.__version__}  "
      f"cuda_available={torch.cuda.is_available()}")
assert transformers.__version__.startswith("4.45"), \
    "SVD-LLM's SVDOPTDecoderLayer only runs on transformers 4.45.x"
PY
}

do_doctor() {
  setup_env
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  log "GPU:"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
    || die "nvidia-smi failed -- this script needs a CUDA GPU"
  "$VENV/bin/python" - <<'PY'
import os, sys, torch, transformers
sys.path.insert(0, os.path.join(os.getcwd(), "local"))
sys.path.insert(0, os.getcwd())
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
print(f"  transformers {transformers.__version__}")
import shared_eval
from SVDLLM import whitening, profle_svdllm, profle_svdllm_low_resource
from utils.data_utils import get_calib_train_data
print("  imports OK (shared_eval + SVD-LLM)")
PY
  log "free space at $SPILL_ROOT:"; df -h "$(dirname "$SPILL_ROOT")" | tail -1
  log "doctor passed"
}

run_one() {
  local model="$1" rho="$2"
  [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV -- run ./run_local.sh setup"
  local tag="${model##*/}_rho$(rho_tag "$rho")"
  local out="$RUNS_DIR/${tag}.json"
  local logf="$LOG_DIR/${METHOD}_${tag}.log"
  mkdir -p "$RUNS_DIR" "$LOG_DIR"

  # (1) layer-streamed profiling. The reference runs used it for everything
  #     except opt-125m; profle_svdllm puts the WHOLE model on the GPU and
  #     accumulates a Gram for EVERY Linear at once (opt-6.7b: 26.6 GB of fp32
  #     weights + 45 GB of Grams = ~72 GB before activations, which OOM'd an
  #     80 GB H100).
  local lr=()
  case "${LOW_RESOURCE:-auto}" in
    1|yes|true) lr=(--low-resource) ;;
    0|no|false) lr=() ;;
    *) [[ "$model" != *125m* ]] && lr=(--low-resource) ;;
  esac

  # (2) spill the per-layer Cholesky factors to disk. Byte-exact; it changes
  #     nothing except peak host RAM. Past 13B, keep the host weight copy in the
  #     checkpoint's own dtype too -- the COMPUTE stays fp32 because
  #     profle_svdllm_low_resource upcasts each layer on its way to the GPU and
  #     narrows it back, which is lossless in both directions (verified
  #     bit-identical on opt-125m).
  local spill=() spill_dir=""
  local want_spill=0
  case "${SPILL:-auto}" in
    1|yes|true) want_spill=1 ;;
    0|no|false) want_spill=0 ;;
    *) [[ "$model" == *13b* || "$model" == *30b* || "$model" == *66b* || "$model" == *70B* ]] && want_spill=1 ;;
  esac
  if (( want_spill )); then
    spill_dir="$SPILL_ROOT/${tag}_$$"
    spill=(SVDLLM_SPILL_DIR="$spill_dir")
    [[ "$model" != *13b* ]] && spill+=(SVDLLM_HOST_DTYPE=source)
  fi

  echo "=========================================================="
  echo "[run_local] method=$METHOD model=$model rho=$rho"
  echo "[run_local] out=$out"
  echo "[run_local] log=$logf"
  echo "[run_local] profiling=$([[ ${#lr[@]} -gt 0 ]] && echo layer-streamed || echo whole-model)"
  [[ -n "$spill_dir" ]] && echo "[run_local] spill=$spill_dir ${spill[*]:1}"
  [[ "${SHARED_EVAL_LAYERWISE:-}" == "1" ]] && echo "[run_local] eval=layerwise"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "=========================================================="

  # `|| rc=$?` rather than a bare call: `set -e` would abort before the cleanup
  # and strand up to 414 GB of spilled factors for every failed run.
  local rc=0
  env "${spill[@]}" "$VENV/bin/python" "$HERE/local/run_svdllm.py" \
      --model "$model" --rho "$rho" --seed "$SEED" \
      --whitening-nsamples 256 --out "$out" "${lr[@]}" ${EXTRA_ARGS:-} \
      2>&1 | tee "$logf" || rc=$?
  [[ -z "$spill_dir" ]] || rm -rf "$spill_dir"
  return $rc
}

do_sweep() {
  setup_env
  local models=("$@"); [[ ${#models[@]} -eq 0 ]] && read -r -a models <<< "$MODELS"
  local n=0 fail=0
  for m in "${models[@]}"; do
    for r in $RHOS; do
      n=$((n+1))
      if run_one "$m" "$r"; then log "OK   $m rho=$r"
      else fail=$((fail+1)); log "FAIL $m rho=$r -- continuing"; fi
    done
  done
  log "sweep done: $((n-fail))/$n succeeded"
  (( fail == 0 ))
}

do_collect() {
  "${VENV}/bin/python" - "$RUNS_DIR" <<'PY'
import csv, glob, json, os, re, sys
root = sys.argv[1]
CANON = re.compile(r"^[A-Za-z0-9._-]+_rho\d+\.json$")
rows, skipped = [], []
for f in sorted(glob.glob(os.path.join(root, "*.json"))):
    if not CANON.match(os.path.basename(f)):
        skipped.append(os.path.basename(f)); continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    # Non-empty whitening_degraded means some matrices fell back to unwhitened
    # SVD because no usable whitening transform existed. Such a row is NOT plain
    # SVD-LLM and must carry the asterisk wherever it is plotted.
    deg = len(d.get("whitening_degraded") or [])
    for corpus, ppl in (d.get("ppl") or {}).items():
        rows.append({
            "method": d.get("method"), "model": d.get("model"),
            "rho": d.get("rho_target"), "dense": d.get("dense"),
            "corpus": corpus, "ppl": ppl,
            "ppl_tokens": (d.get("ppl_tokens") or {}).get(corpus),
            "whitening_degraded_matrices": deg,
            "realised_linear_ratio": d.get("realised_linear_ratio"),
            "realised_total_ratio": d.get("realised_total_ratio"),
            "compress_seconds": d.get("compress_seconds"), "source": f,
        })
if not rows:
    print("no results yet"); raise SystemExit(0)
rows.sort(key=lambda r: (r["model"] or "", str(r["rho"]), r["corpus"]))
out = os.path.join(root, "results.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} row(s) -> {out}")
if skipped:
    print(f"skipped {len(skipped)} non-canonical file(s): {', '.join(skipped)}")
print()
print(f"{'model':<24}{'rho':>6}  {'corpus':<11}{'ppl':>12}  {'degraded':>9}")
for r in rows:
    print(f"{(r['model'] or '').split('/')[-1]:<24}"
          f"{(r['rho'] if r['rho'] is not None else 'dense'):>6}  "
          f"{r['corpus']:<11}{r['ppl']:>12.4f}  "
          f"{r['whitening_degraded_matrices']:>9}")
PY
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  setup)   do_setup "$@" ;;
  doctor)  do_doctor ;;
  run)     [[ $# -eq 2 ]] || die "usage: ./run_local.sh run MODEL RHO"
           setup_env; run_one "$1" "$2" ;;
  sweep)   do_sweep "$@" ;;
  collect) do_collect ;;
  help|-h|--help) sed -n '2,70p' "$0" | sed 's/^# \?//' ;;
  *) die "unknown command: $cmd (try ./run_local.sh help)" ;;
esac
