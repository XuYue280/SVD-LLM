# `local/` — single-GPU runner support files

Everything here exists so `../run_local.sh` can run this baseline on an ordinary
GPU box (e.g. Lambda Labs) with no Slurm and no cluster-specific paths.

| file | origin | change |
|---|---|---|
| `run_svdllm.py` | ARKS comparison harness, `baselines/tools/run_svdllm.py` | repo root resolved from `__file__` instead of a hard-coded `/scratch` path |
| `shared_eval.py` | ARKS comparison harness, `baselines/tools/shared_eval.py` | `LOCAL_FILES` made environment-driven (`SHARED_EVAL_PTB_FILE`, `SHARED_EVAL_C4_FILE`) instead of hard-coded cluster paths |

## Single-GPU / layerwise / backward — the short answer

Of the three SVD baselines this is the one that is already almost entirely
single-GPU and layerwise.

* **Backward pass: none on this path.** `whitening_local_update`,
  `local_update` and `fasterprune` are all `@torch.no_grad()` closed-form
  updates, not training. The only real backward in the repo is `utils/LoRA.py`
  (HF `Trainer`, whole model on GPU), which this runner never invokes.
* **Profiling: layerwise, natively.** `profle_svdllm_low_resource` is upstream's
  own layer-streaming implementation — the model stays on the host and only the
  embeddings/norms plus one decoder layer are on the GPU at a time. It is the
  default here for everything except opt-125m. The alternative,
  `profle_svdllm`, does `model.to(dev)` and needs the whole model resident.
* **Whitening + truncated SVD: per-matrix**, finer than layerwise. Size the card
  against four `d_in x d_in` fp64 buffers, not one — the input and the
  out-of-place Cholesky factor are both live and the escalating-ridge retry adds
  two more (~6.6 GB Llama-3.1-8B, ~13.4 GB opt-13b, ~26.3 GB opt-30b).
* **Host RAM** is the other limit: the per-layer Cholesky factors would
  otherwise all stay resident (176 GB on opt-13b, 414 GB on opt-30b).
  `SVDLLM_SPILL_DIR` pages them to disk; it is byte-exact and changes no number.
* **Evaluation:** whole-model by default, `LAYERWISE_EVAL=1` streams decoder
  layers — needed for models that do not fit one card.

See the header of `../run_local.sh` for the exact conditionals.
