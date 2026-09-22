#!/usr/bin/env python3
"""Compress with SVD-LLM (whitening), then evaluate through the shared harness.

Ratio convention: SVDLLM.py:507's help text claims --ratio is the fraction
KEPT, but SVDLLM.py:523 does `args.ratio = 1 - args.ratio` before use, so the
CLI value is the fraction REMOVED. This script takes --rho (fraction KEPT, the
ARKS convention) and passes `rho` straight to whitening(), which is the
already-inverted value the internals expect. No double inversion.

SVD-LLM's whitening() walks model.model.decoder.layers with find_layers(), so
it touches decoder linears only -- lm_head is left alone, matching ARKS.
"""
# VENDORED COPY -- the repo root is resolved from this file's location
# instead of the hard-coded /scratch path the Trillium harness used, so
# this runs unchanged from any checkout directory.
import argparse, json, os, sys, time
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)          # <checkout>/local/..  ->  <checkout>
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

from shared_eval import (evaluate_all, count_parameters, arks_fields,
                         PeakMemory, dense_cache_meta,
                         load_dense_cache, save_dense_cache)

# Upstream's value (utils/model_utils.py:21); the eval recipe uses 2048 too.
SEQLEN = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rho", type=float, required=True, help="fraction of params KEPT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-dataset", default="wikitext2")
    ap.add_argument("--whitening-nsamples", type=int, default=256)
    # opt-30b needs ~1h35m of profiling plus ~1h20m of whitening -- 3 h against
    # debug's 2 h ceiling, and compute queues for days. profiling_mat is already
    # spilled to disk byte-exactly, so the run splits there at no numerical cost:
    #   stage=profile  profile, keep the spill dir, exit
    #   stage=whiten   skip profiling, load the spill dir, compress and evaluate
    # Requires a STABLE --spill-dir shared by the two jobs.
    ap.add_argument("--stage", default="all", choices=["all", "profile", "whiten"])
    ap.add_argument("--spill-dir", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low-resource", action="store_true")
    ap.add_argument("--dense", action="store_true")
    a = ap.parse_args()

    import random, numpy as np
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # SVD-LLM's profiling/whitening code assumes every tensor is on one device;
    # device_map="auto" shards across the visible GPUs and blows up with
    # "Expected all tensors to be on the same device". Load on a single GPU, or
    # keep it on CPU and let the low-resource path stream layers to the GPU.
    # float32, not fp16: whitening() runs a Cholesky decomposition and builds the
    # U/V factors in fp32. Loading fp16 leaves the untouched layers half and the
    # new factors float -> "expected mat1 and mat2 to have the same dtype".
    # Everything is cast back to fp16 after compression, before evaluation.
    # SVDLLM_HOST_DTYPE=source keeps the HOST copy in the checkpoint's own dtype
    # (fp16 for opt-30b/66b, bf16 for Llama-3.1-70B) instead of fp32:
    #     opt-30b  120 GB -> 60 GB      opt-66b 263 GB -> 131 GB
    # against a 201 GB cgroup. This is not optional past 30B: job 960441 was
    # OOM-killed at layer 15/48 on opt-30b, because 120 GB of fp32 weights plus
    # the page cache from spilling 8.63 GB of Cholesky factors per layer reached
    # the cap.
    #
    # The COMPUTE stays fp32: profle_svdllm_low_resource upcasts each layer on
    # its way to the GPU and narrows it back on the way out. Widening fp16 -> fp32
    # is lossless and narrowing recovers the same bits, so the calibration
    # forward, the Grams and the SVD are bit-identical to an fp32 load -- verified
    # on opt-125m (job 958923: 72.0123176845 / 356.9292006203 / 165.3395169235,
    # zero deviation).
    #
    # low_cpu_mem_usage=True matters independently: without it from_pretrained
    # random-initialises a full model before overwriting it shard by shard, so the
    # peak is the model twice over.
    _host_src = os.environ.get("SVDLLM_HOST_DTYPE", "") == "source"
    dt = torch.float32 if not a.dense else torch.float16
    _kw = dict(torch_dtype=("auto" if _host_src else dt), low_cpu_mem_usage=True)
    if a.low_resource:
        model = AutoModelForCausalLM.from_pretrained(a.model, **_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, **_kw).to("cuda")
    # SVD-LLM reads model.seqlen (profle_svdllm_low_resource sizes its `inps`
    # buffer from it). It is not a HuggingFace attribute -- upstream stamps it in
    # utils/model_utils.py:21 get_model_from_huggingface, which this driver
    # bypasses by calling AutoModelForCausalLM directly. opt-125m never caught it
    # because run.sh only passes --low-resource for *13b*/*30b*.
    model.seqlen = SEQLEN
    model.eval()
    # ARKS bench-record timings / peak memory (shared_eval.PeakMemory)
    _t_wall0 = time.perf_counter()
    _mem = PeakMemory(); _mem.__enter__()
    _t_load = time.perf_counter() - _t_wall0

    before = count_parameters(model)
    t0 = time.perf_counter()

    if not a.dense:
        import SVDLLM as _svdllm
        from SVDLLM import whitening, profle_svdllm, profle_svdllm_low_resource
        from utils.data_utils import get_calib_train_data

        calib = get_calib_train_data(a.calib_dataset, tok, a.whitening_nsamples,
                                     seqlen=SEQLEN, seed=a.seed)
        if a.spill_dir:
            os.environ["SVDLLM_SPILL_DIR"] = a.spill_dir
        if a.stage == "whiten":
            # Rebuild the mapping from what the profile stage left on disk. The
            # files hold exactly the tensors profiling produced -- torch.save /
            # torch.load is byte-exact -- so resuming here is identical to having
            # run both stages in one process.
            import SVDLLM as _sv
            prof = _sv._SpillingProfilingMat(a.spill_dir)
            n_layers = len([f for f in os.listdir(a.spill_dir)
                            if f.startswith("layer_") and f.endswith(".pt")])
            if not n_layers:
                raise SystemExit(f"--stage whiten: no layer_*.pt in {a.spill_dir}")
            for _i in range(n_layers):
                _p = os.path.join(a.spill_dir, f"layer_{_i}.pt")
                if not os.path.exists(_p):
                    raise SystemExit(f"--stage whiten: missing {_p}")
                dict.__setitem__(prof, _i, _p)
            print(f"[stage] resumed {n_layers} layer profiles from {a.spill_dir}",
                  flush=True)
        else:
            prof = (profle_svdllm_low_resource if a.low_resource else profle_svdllm)(
                a.model, model, calib, "cuda")
            if a.stage == "profile":
                print(f"[stage] profiling done, {len(prof)} layers in "
                      f"{a.spill_dir or '(memory)'}", flush=True)
                raise SystemExit(0)
        # `a.rho` is the KEEP fraction; whitening() expects exactly that.
        whitening(a.model, model, prof, a.rho, "cuda")
        del prof
        torch.cuda.empty_cache()
        # profle_svdllm ends with an in-place model.cpu() (SVDLLM.py ~line 51),
        # so the model comes back on CPU regardless of where it started.
        model = model.to("cuda")

    compress_s = time.perf_counter() - t0
    after = count_parameters(model)
    # ARKS: layers_evaluated / matrices_evaluated (bench record :1043-1044)
    _n_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    _n_matrices = sum(1 for _n, _m in model.named_modules()
                      if isinstance(_m, torch.nn.Linear) and "lm_head" not in _n)
    # Unconditional: after whitening the model is a mix of fp16 (untouched) and
    # fp32 (new factors), so checking only the first parameter is not enough.
    model = model.half().to("cuda")
    _t_eval0 = time.perf_counter()
    res = evaluate_all(model, tok, device="cuda")
    _t_eval1 = time.perf_counter()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    payload = {
        # Non-empty means some matrices fell back to unwhitened SVD because no
        # usable whitening transform existed. Such a row is NOT plain SVD-LLM and
        # must be labelled as degraded wherever it is plotted or tabulated.
        "whitening_degraded": list(getattr(__import__("SVDLLM"), "WHITENING_DEGRADED", [])) if not a.dense else [],
        "method": "svdllm", "model": a.model,
        "rho_target": None if a.dense else a.rho, "dense": a.dense, "seed": a.seed,
        "calib_dataset": a.calib_dataset, "whitening_nsamples": a.whitening_nsamples,
        "params_before": before, "params_after": after,
        "realised_total_ratio": after["total_params"] / before["total_params"],
        "realised_linear_ratio": after["linear_params"] / before["linear_params"],
        "compress_seconds": compress_s,
        "ppl": {c: v["ppl"] for c, v in res.items()},
        "ppl_tokens": {c: v["ppl_tokens"] for c, v in res.items()},
    }
    # --- ARKS-compatible fields (see shared_eval.arks_fields) ---
    _runs_root = os.path.dirname(a.out)
    if a.dense:
        save_dense_cache(_runs_root, res, inference_seconds=_t_eval1 - _t_eval0)
    _dense_res = load_dense_cache(_runs_root)
    payload.update(arks_fields(
        method="svdllm", model=a.model, rho=(None if a.dense else a.rho),
        res=res, params_before=before, params_after=after,
        dense=a.dense, seed=a.seed, dense_res=_dense_res,
        timings={
            "inference_seconds": _t_eval1 - _t_eval0,
            "metric_seconds": _t_eval1 - _t_eval0,
            "model_load_seconds": _t_load,
            "reconstruction_seconds": compress_s if not a.dense else 0.0,
            "artifact_load_seconds": 0.0,
            "dense_inference_seconds": (_dense_res.get("_meta") or {}).get("dense_inference_seconds"),
            "total_seconds": time.perf_counter() - _t_wall0,
        },
        peak_memory=_mem.fields(),
        layers_evaluated=_n_layers, matrices_evaluated=_n_matrices))

    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\n" + json.dumps(payload["ppl"], indent=2))
    print(f"realised linear ratio = {payload['realised_linear_ratio']:.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
