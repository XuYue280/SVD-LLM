"""Perplexity evaluation shared by all baselines, byte-compatible with ARKS.

Why this exists
---------------
ASVD, Basis_Sharing and SVD-LLM each ship their own perplexity harness, and all
three differ from the ARKS/GPTQ convention in ways that change the number:

  ASVD          PTB uses the *validation* split and "\\n\\n".join  -> dense 32.55
  Basis_Sharing PTB "\\n\\n".join without strip; C4 test=val, rows [0:2000], no cap
  SVD-LLM       PTB "\\n\\n".join

Comparing those against ARKS would be comparing different measurements. Rather
than patching three codebases into agreement (and hoping they stay in
agreement), every baseline hands its compressed model to this one module.
Comparability is then structural, not a property anyone has to maintain.

The recipes and the perplexity reduction below are transcribed from
final1/arks_common.py (EVAL_RECIPES) and
final1/bench_inference_minibatch_vq.py (test_loader / perplexity).

Reference, facebook/opt-125m dense fp16 seqlen 2048:
    wikitext2 27.65    ptb 38.99    c4 26.56
"""
from __future__ import annotations

import json
import math
import os

import torch

SEQLEN = 2048

# name -> (hf id, config, split, data_files, separator, strip_rows, max_rows, max_chunks)
RECIPES = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", None,
                  "\n\n", False, None, 0),
    # The `text` builder over the original Penn Treebank files: one line per row.
    # Rows must be stripped before joining -- unstripped gives 30.22, not 38.98.
    # The raw githubusercontent URL is unreachable from a compute node, so a
    # local copy is preferred when present (see _local_or_remote).
    "ptb": ("text", None, "test",
            "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
            " ", True, None, 0),
    "c4": ("allenai/c4", "en", "validation",
           "en/c4-validation.00000-of-00008.json.gz",
           " ", False, 1100, 256),
}

# A machine with no outbound network cannot fetch what a recipe names remotely.
# Where that happens, fall back to a local copy so the same code runs online and
# offline without branching.
# corpus -> (builder, local path). The builder differs from the online recipe
# because we bypass the hub: "allenai/c4" + a repo-relative shard name resolves
# through the hub and its cache key does not match what the shard was cached
# under, whereas the "json" builder reads the very same .json.gz directly.
# Set these when the machine has no outbound network, pointing at local copies
# of the two files the online recipes fetch. Row content is identical either
# way, so the tokenised corpus -- and the perplexity -- is unchanged.
#   SHARED_EVAL_PTB_FILE  ptb.test.txt
#                         (raw.githubusercontent.com/wojzaremba/lstm/master/data/)
#   SHARED_EVAL_C4_FILE   en/c4-validation.00000-of-00008.json.gz
#                         (from the allenai/c4 hub repo)
# Unset, or pointing at something that does not exist, means the online recipe
# in RECIPES above is used.
LOCAL_FILES = {
    "ptb": ("text", os.environ.get("SHARED_EVAL_PTB_FILE", "")),
    "c4": ("json", os.environ.get("SHARED_EVAL_C4_FILE", "")),
}


def build_text(corpus: str, tokenizer):
    """Tokenise one corpus into a flat id tensor under the ARKS recipe."""
    from datasets import load_dataset

    name, config, split, files, sep, strip_rows, max_rows, _ = RECIPES[corpus]
    kw = {"verification_mode": "no_checks"}
    local = LOCAL_FILES.get(corpus)
    if local and os.path.exists(local[1]):
        # Offline path: read the file directly. Row content is identical to the
        # online recipe, so the tokenised corpus (and the perplexity) matches.
        ds = load_dataset(local[0], data_files={split: local[1]}, split=split, **kw)
    elif files:
        ds = load_dataset(name, config, split=split, data_files={split: files}, **kw)
    else:
        ds = load_dataset(name, config, split=split, **kw)
    col = "text" if "text" in ds.column_names else ds.column_names[0]
    rows = list(ds[col])
    if max_rows is not None:
        rows = rows[:max_rows]
    if strip_rows:
        rows = [r.strip() for r in rows]
    return tokenizer(sep.join(rows), return_tensors="pt").input_ids[0]


class _StopForward(Exception):
    """Raised by the catcher to abandon the forward once layer 0 is reached."""


class _LayerInputCatcher(torch.nn.Module):
    """Records the hidden state and kwargs layer 0 is called with, then stops."""

    def __init__(self, inner, sink):
        super().__init__()
        self.inner = inner
        self._sink = sink

    def forward(self, hidden, **kwargs):
        self._sink.append((hidden, kwargs))
        raise _StopForward


def _decoder_stack(model):
    """(layers, tail) for the OPT and Llama families.

    `tail(h)` applies everything between the last decoder layer and the LM head,
    so `lm_head(tail(h))` equals what `model(...).logits` would have produced.
    Transcribed from ARKS final1/bench_inference_minibatch_vq.py:496 so the two
    harnesses stay one implementation apart, not two.
    """
    inner = model.model
    if hasattr(inner, "decoder"):  # OPT
        decoder = inner.decoder

        def tail(hidden):
            if getattr(decoder, "final_layer_norm", None) is not None:
                hidden = decoder.final_layer_norm(hidden)
            if getattr(decoder, "project_out", None) is not None:
                hidden = decoder.project_out(hidden)
            return hidden

        return decoder.layers, tail

    def tail(hidden):  # Llama and friends
        return inner.norm(hidden)

    return inner.layers, tail


def _set_decoder_layers(model, layers):
    inner = model.model
    if hasattr(inner, "decoder"):
        inner.decoder.layers = layers
    else:
        inner.layers = layers


def _nll_from_hidden(model, hidden, labels, loss_fn, chunk_tokens):
    """Token NLL from the decoder output, matching perplexity() exactly."""
    if chunk_tokens and chunk_tokens > 0:
        positions = labels.shape[1]
        nll = torch.empty(labels.shape, dtype=torch.float32, device=labels.device)
        for a in range(0, positions, chunk_tokens):
            b = min(a + chunk_tokens, positions)
            logits = model.lm_head(hidden[:, a:b, :])
            nll[:, a:b] = loss_fn(logits.permute(0, 2, 1), labels[:, a:b]).float()
            del logits
        return nll
    # Slice AFTER the head, not before: lm_head(h)[:, :-1] and lm_head(h[:, :-1])
    # are the same maths but different matmul shapes, and a different shape can
    # pick a different cuBLAS kernel -- which would break bit-identity with the
    # monolithic path for no benefit.
    logits = model.lm_head(hidden)[:, :-1, :]
    return loss_fn(logits.permute(0, 2, 1), labels)


@torch.no_grad()
def _perplexity_layerwise(model, x, device, pad_token_id, loss_fn, batch_size,
                          chunk_tokens):
    """perplexity() with ONE decoder layer resident on the device at a time.

    The model stays on the host; each decoder layer is moved to `device`, used
    for every sample, then moved back. Peak VRAM is one layer plus activations
    instead of the whole model:

        opt-66b       131.44 GB of fp16 weights -> ARKS measures 13.2/6.6/21.3 GB
        Llama-3.1-70B 141.11 GB                 -> ARKS measures 15.8/9.2/23.4 GB

    (those are peak_cuda_gb from ARKS's results_summary.csv for wikitext2/ptb/c4
    on one H100 -- measured, not projected.)

    Loop order is layer-outer, sample-inner ON PURPOSE: each layer then crosses
    the bus once per evaluation rather than once per sample. The opposite order
    costs (samples x model size) of transfers and is unusable past ~30B.

    Every sample still goes through the same modules, in the same dtype, on the
    same device as the monolithic path, so the result matches bit for bit rather
    than merely closely. Transcribed from ARKS
    final1/bench_inference_minibatch_vq.py:550.
    """
    layers, tail = _decoder_stack(model)
    use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    try:
        # Everything except the decoder layers stays resident: embeddings,
        # rotary helpers, the final norm and the head are all small.
        _set_decoder_layers(model, torch.nn.ModuleList())
        model.to(device)
        _set_decoder_layers(model, layers)

        # Pass 1: replay the real forward up to layer 0, so the per-sample kwargs
        # (attention mask, position ids, position embeddings, cache position) are
        # exactly what the model itself would have passed -- not reconstructed.
        sink = []
        layers[0] = _LayerInputCatcher(layers[0], sink)
        batches = []
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size].to(device)
            batches.append(batch)
            try:
                model(input_ids=batch, attention_mask=torch.ones_like(batch))
            except _StopForward:
                pass
        layers[0] = layers[0].inner
        if not sink:
            raise ValueError("layerwise: no samples captured at layer 0")

        states = [h for h, _ in sink]
        kwargs_per_sample = [k for _, k in sink]

        # Pass 2: one layer at a time, every sample through it.
        for layer in layers:
            layer.to(device)
            for i, hidden in enumerate(states):
                out = layer(hidden, **kwargs_per_sample[i])
                states[i] = out[0] if isinstance(out, tuple) else out
            layer.to("cpu")
            if device == "cuda" or getattr(device, "type", "") == "cuda":
                torch.cuda.empty_cache()

        # Pass 3: final norm (+ OPT's project_out) and the LM head.
        total_nll, total_tokens = 0.0, 0
        for batch, hidden in zip(batches, states):
            labels = batch[:, 1:]
            nll = _nll_from_hidden(model, tail(hidden), labels, loss_fn, chunk_tokens)
            mask = (labels != pad_token_id) if pad_token_id is not None \
                else torch.ones_like(labels, dtype=torch.bool)
            total_nll += float((nll * mask).sum())
            total_tokens += int(mask.sum())
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
    return total_nll, total_tokens


@torch.no_grad()
def perplexity(model, ids, device, pad_token_id, max_chunks=0, batch_size=1):
    """exp(sum NLL / total predicted tokens), identical to ARKS's reduction."""
    n = ids.numel() // SEQLEN
    x = ids[: n * SEQLEN].reshape(n, SEQLEN)
    if max_chunks:
        x = x[:max_chunks]
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="none", ignore_index=pad_token_id if pad_token_id is not None else -100
    )
    model.eval()
    # SHARED_EVAL_LAYERWISE=1 streams decoder layers instead of holding the whole
    # model on the GPU -- required for opt-66b (131 GB fp16) and Llama-3.1-70B
    # (141 GB), which do not fit on one 80 GB H100. Unset = the monolithic path
    # every validated number was produced with.
    if os.environ.get("SHARED_EVAL_LAYERWISE", "") == "1":
        chunk = int(os.environ.get("SHARED_EVAL_LOSS_CHUNK_TOKENS", "0"))
        total_nll, total_tokens = _perplexity_layerwise(
            model, x, device, pad_token_id, loss_fn, batch_size, chunk)
        if not total_tokens:
            raise ValueError("no tokens evaluated")
        return math.exp(total_nll / total_tokens), total_tokens
    total_nll, total_tokens = 0.0, 0
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size].to(device)
        logits = model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits[:, :-1, :]
        labels = batch[:, 1:]
        nll = loss_fn(logits.permute(0, 2, 1), labels)
        mask = (labels != pad_token_id) if pad_token_id is not None \
            else torch.ones_like(labels, dtype=torch.bool)
        total_nll += float((nll * mask).sum())
        total_tokens += int(mask.sum())
        del logits
    if not total_tokens:
        raise ValueError("no tokens evaluated")
    return math.exp(total_nll / total_tokens), total_tokens


def evaluate_all(model, tokenizer, device="cuda", corpora=("wikitext2", "ptb", "c4"),
                 batch_size=1):
    """Return {corpus: {ppl, ppl_tokens}} for every requested corpus."""
    out = {}
    # A compressed model whose weights went non-finite still runs: every logit is
    # NaN, cross-entropy is NaN, and the job exits 0 having written a result file
    # full of NaN. That happened to SVD-LLM on opt-6.7b (jobs 942330_0/_1, both
    # "COMPLETED"). A silently invalid number in the results table is worse than a
    # failed job, so refuse to return one -- the caller must fail loudly instead.
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(
            f"{len(bad)} parameter tensor(s) are non-finite before evaluation, "
            f"first: {bad[:3]}. The compression produced an unusable model; "
            f"refusing to emit perplexities."
        )
    for corpus in corpora:
        ids = build_text(corpus, tokenizer)
        cap = RECIPES[corpus][7]
        ppl, ntok = perplexity(model, ids, device, tokenizer.pad_token_id, cap, batch_size)
        if not math.isfinite(ppl):
            raise RuntimeError(
                f"{corpus} perplexity is {ppl} with finite weights -- evaluation "
                f"itself diverged; refusing to emit it."
            )
        out[corpus] = {"ppl": ppl, "ppl_tokens": ntok}
        print(f"[eval] {corpus:<10} ppl={ppl:.4f} tokens={ntok}", flush=True)
    return out


def count_parameters(model):
    """Total and decoder-linear parameter counts, for reporting the realised ratio."""
    total = sum(p.numel() for p in model.parameters())
    linear = sum(m.weight.numel() for m in model.modules()
                 if isinstance(m, torch.nn.Linear) and hasattr(m, "weight"))
    return {"total_params": int(total), "linear_params": int(linear)}


def write_result(path, *, method, model, ratio, corpus, ppl, ppl_tokens,
                 dense_ppl=None, extra=None):
    """Append one row in the ARKS results schema (plus method/ratio columns)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {
        "method": method, "model": model, "ratio": ratio, "corpus": corpus,
        "ppl": ppl, "ppl_tokens": ppl_tokens, "dense_ppl": dense_ppl,
        "ppl_increase_percent": (100.0 * (ppl / dense_ppl - 1.0)) if dense_ppl else None,
    }
    if extra:
        rec.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    return rec
