#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)



# Matrices for which no usable whitening transform could be derived, so plain SVD
# was used instead. run_svdllm.py copies this into the result JSON: a degraded run
# is still a number, but it is not the same method and must not be reported as if
# it were.
WHITENING_DEGRADED = []


class _SpillingProfilingMat(dict):
    """Per-layer whitening factors, spilled to disk instead of held in RAM.

    profle_svdllm_low_resource builds a float64 Cholesky factor for every Linear
    in every layer and keeps them ALL resident, while whitening() consumes them
    strictly one layer at a time. The resident set is
    (sum over Linears of d_in^2) * 8 bytes * n_layers:

        opt-6.7b      2.82 GB/layer x 32 =  90 GB   fits
        Llama-3.1-8B  2.45 GB/layer x 32 =  78 GB   fits
        opt-13b       4.40 GB/layer x 40 = 176 GB   OOM-killed (942287_2/_3,
                                                    MaxRSS 187.5 GiB = the cap)
        opt-30b       8.63 GB/layer x 48 = 414 GB

    Trillium allots 7.8125 GiB per core and caps a 1-GPU job at 24 cores, so the
    cgroup is 187.5 GiB and cannot be raised without taking a whole 4-GPU node.
    Storing float32 instead would halve it but would run torch.linalg.inv on a
    badly conditioned matrix in single precision, so that is not an option.

    Round-tripping through torch.save/torch.load is byte-exact, so this changes
    no number; it only trades RAM for scratch I/O. One layer is cached because
    whitening() indexes profiling_mat[i][name] once per Linear within a layer.
    """

    def __init__(self, root):
        super().__init__()
        self._root = root
        self._cache_idx = None
        self._cache_val = None
        os.makedirs(root, exist_ok=True)

    def __setitem__(self, i, layer_profile):
        path = os.path.join(self._root, f"layer_{i}.pt")
        torch.save(layer_profile, path)
        # Drop the WRITE-BACK cache too, not just the read cache. Profiling only
        # ever writes, so a DONTNEED on __getitem__ alone never runs during the
        # stage that actually fills the cache: job 961395 wrote 330 GB across 38
        # layers and grew 3.3 GB/layer instead of the 1.23 GB/layer the converted
        # weights account for, heading for the 187.5 GiB cap at about layer 44.
        # fsync first -- DONTNEED silently skips dirty pages, so without it this
        # is a no-op on exactly the pages that matter.
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:
            pass
        super().__setitem__(i, path)

    def __getitem__(self, i):
        if self._cache_idx == i:
            return self._cache_val
        path = super().__getitem__(i)
        val = torch.load(path, map_location="cpu")
        # Drop this file's page cache immediately. The whole point of spilling is
        # to trade RAM for disk, but the kernel keeps every byte read back in the
        # cache, which counts against the cgroup: opt-30b reads 145 GB of factors
        # during whitening and job 960798 was OOM-killed doing exactly that.
        # DONTNEED only evicts clean pages -- the data stays on disk, so a re-read
        # still works, just from the device.
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:
            pass
        self._cache_idx, self._cache_val = i, val
        return val


def _new_profiling_mat():
    """Disk-backed when SVDLLM_SPILL_DIR is set, otherwise upstream's plain dict."""
    root = os.environ.get("SVDLLM_SPILL_DIR", "")
    return _SpillingProfilingMat(root) if root else {}


def _spd_cholesky(raw, dev):
    """Cholesky of a Gram matrix that upstream's absolute 1e-6 ridge cannot rescue.

    Called ONLY after upstream's own `(-eigenvalues[0] + 1e-6) * I` shift has been
    tried and its cholesky still raised, so runs where the shipped path works
    (opt-125m takes it 8 times and completes) are bit-for-bit unaffected.

    The shipped ridge is ABSOLUTE. These Grams are sums over ~522k tokens with
    ||G|| ~ 1e13, and OPT's LayerNorms all have gamma == 1 exactly, which pins
    1^T y to a constant and leaves the all-ones direction with a Rayleigh
    quotient many orders below lambda_max. A fixed 1e-6 is then far below the
    fp32 accumulation noise it is meant to dominate, so it neither restores
    definiteness nor bounds the condition number. Scale the ridge to the matrix
    instead, escalating until the factor is finite.
    """
    if not torch.isfinite(raw).all():
        # A non-finite Gram cannot be repaired by any ridge (NaN + x = NaN), and
        # torch.linalg.cholesky does not always reject it -- it can "succeed" and
        # return a factor containing Inf, which then makes inv/pinv produce NaN
        # and every downstream weight NaN. Drop the poisoned entries first.
        n_bad = int((~torch.isfinite(raw)).sum())
        print(f"Warning: Gram has {n_bad} non-finite entries; zeroing them")
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    scale = torch.diagonal(raw).abs().max().clamp(min=1.0)
    eye = torch.eye(raw.shape[0], dtype=raw.dtype, device=dev)
    for rel in (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        try:
            L = torch.linalg.cholesky(raw + (rel * scale) * eye)
        except Exception:
            continue
        if torch.isfinite(L).all():
            print(f"Info: relative ridge {rel:g}*{scale:.3e} restored definiteness")
            return L
    raise RuntimeError("scaling_diag_matrix is not salvageable by a relative ridge")


@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = _new_profiling_mat()
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                # Belt-and-braces. Measured on this torch build (n=2048 probes):
                # eigvalsh RAISES on a non-finite input, and RAISES "failed to
                # converge" on an ill-conditioned one -- it does NOT return NaN
                # silently, so this branch may never fire. Kept because if it ever
                # did, `raw += (-NaN + 1e-6) * I` would poison the whole diagonal
                # of a Gram that was itself fine, and that is unrecoverable.
                # Upstream calls eigvalsh inside the except block with no guard of
                # its own, so a raise propagates and kills the job -- left as is,
                # because loud beats silent.
                # if/else rather than an early `continue`: the loop body ends with
                # `subset[name].raw_scaling_diag_matrix = None; del ...;
                # empty_cache()`, which releases the per-module Gram (1.7 GB for a
                # single opt-13b fc2). Skipping that leaks it for every matrix.
                if not torch.isfinite(eigenvalues).all():
                    print("Warning: eigvalsh returned non-finite eigenvalues; "
                          "keeping the original Gram and ridging it directly")
                    scaling_diag_matrix = _spd_cholesky(raw_scaling_diag_matrix, dev)
                else:
                    raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                    try:
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                    except Exception:
                        # upstream leaves this second cholesky unguarded; on opt-6.7b+
                        # the absolute 1e-6 shift is too small to make it succeed
                        scaling_diag_matrix = _spd_cholesky(raw_scaling_diag_matrix, dev)
                    if not torch.isfinite(scaling_diag_matrix).all():
                        # cholesky returned WITHOUT raising but with Inf/NaN entries.
                        # CONFIRMED on this build: given a matrix with a single Inf
                        # on the diagonal, torch.linalg.cholesky does not raise and
                        # returns a non-finite factor (eigvalsh raises on the same
                        # input). Unchecked, this is what made SVD-LLM write an all-NaN result
                        # for opt-6.7b while exiting 0 (jobs 942330_0/_1).
                        print("Warning: cholesky returned a non-finite factor; re-deriving")
                        scaling_diag_matrix = _spd_cholesky(raw_scaling_diag_matrix, dev)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    # opt-66b is 263 GB in fp32 against a 201 GB cgroup, so its host copy must be
    # fp16. Compute still happens in fp32: the checkpoint's values ARE fp16, so
    # widening them is lossless and a layer computes bit-identically whether the
    # host copy was stored fp16 or fp32. Every line guarded by `dtype` below is a
    # no-op when the model is already fp32 -- which is every validated run -- so
    # those numbers cannot move.
    _src_dtype = next(iter(model.parameters())).dtype
    dtype = torch.float32 if _src_dtype != torch.float32 else _src_dtype
    if "opt" in model_name:
        layers = model.model.decoder.layers
        # opt-350m is the one OPT variant that breaks both assumptions here:
        # do_layer_norm_before=False makes `final_layer_norm` None
        # (modeling_opt.py builds it only when that flag is set), and
        # word_embed_proj_dim 512 != hidden_size 1024 gives it project_in /
        # project_out, which every other OPT lacks and which upstream therefore
        # never moves -- leaving them on the host while the batch is on the GPU.
        # Both are None on every other OPT, so this is inert for them.
        dec = model.model.decoder
        for _name in ("embed_tokens", "final_layer_norm", "embed_positions",
                      "project_in", "project_out"):
            _mod = getattr(dec, _name, None)
            if _mod is not None:
                setattr(dec, _name, _mod.to(dev, dtype))
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev, dtype)
        model.model.norm = model.model.norm.to(dev, dtype)
        # transformers >= 4.43 hoisted the rotary embedding to a module on
        # LlamaModel (modeling_llama.py:976 calls self.rotary_emb before the
        # decoder loop). This function predates that, so inv_freq stayed on the
        # CPU while position_ids arrived on the GPU:
        #   RuntimeError: Expected all tensors to be on the same device ... line 211
        # OPT has no such module, which is why only the Llama path broke.
        if getattr(model.model, "rotary_emb", None) is not None:
            model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev, dtype)

    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    # 'first' is a separate sentinel: upstream used cache['attention_mask'] is None
    # to mean "first batch", but transformers >= 4.43 legitimately passes
    # attention_mask=None for a plain causal, unpadded batch under SDPA
    # (_update_causal_mask returns None and SDPA gets is_causal=True instead).
    # Llama-3.1-8B hits that path, so the flag never flipped and the branch that
    # calls .cpu() on None ran on every batch -> AttributeError. OPT still
    # materialises a 4-D mask, so its behaviour is unchanged.
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'first': True}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            am = kwargs.get('attention_mask', None)
            pi = kwargs.get('position_ids', None)
            if cache['first']:
                cache['first'] = False
                cache['attention_mask'] = None if am is None else am.cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = None if pi is None else pi.cpu()
            else:
                if am is not None and cache['attention_mask'] is not None:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], am.cpu()), dim=0)
                if "opt" not in model_name and pi is not None and cache['position_ids'] is not None:
                    cache['position_ids'] = torch.cat((cache['position_ids'], pi.cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        dec = model.model.decoder
        for _name in ("embed_tokens", "final_layer_norm", "embed_positions",
                      "project_in", "project_out"):
            _mod = getattr(dec, _name, None)
            if _mod is not None:
                setattr(dec, _name, _mod.cpu())
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        if getattr(model.model, "rotary_emb", None) is not None:
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = _new_profiling_mat()
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i].to(dev, dtype)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            # None means "plain causal, no padding" -- replaying it as None
            # reproduces exactly what the capture pass saw (SDPA is_causal=True).
            am_j = None if attention_masks is None else attention_masks[j].unsqueeze(0).to(dev)
            if "opt" not in model_name:
                pi_j = None if position_ids is None else position_ids[j].unsqueeze(0).to(dev)
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=am_j, position_ids=pi_j)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=am_j)[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                # Belt-and-braces. Measured on this torch build (n=2048 probes):
                # eigvalsh RAISES on a non-finite input, and RAISES "failed to
                # converge" on an ill-conditioned one -- it does NOT return NaN
                # silently, so this branch may never fire. Kept because if it ever
                # did, `raw += (-NaN + 1e-6) * I` would poison the whole diagonal
                # of a Gram that was itself fine, and that is unrecoverable.
                # Upstream calls eigvalsh inside the except block with no guard of
                # its own, so a raise propagates and kills the job -- left as is,
                # because loud beats silent.
                # if/else rather than an early `continue`: the loop body ends with
                # `subset[name].raw_scaling_diag_matrix = None; del ...;
                # empty_cache()`, which releases the per-module Gram (1.7 GB for a
                # single opt-13b fc2). Skipping that leaks it for every matrix.
                if not torch.isfinite(eigenvalues).all():
                    print("Warning: eigvalsh returned non-finite eigenvalues; "
                          "keeping the original Gram and ridging it directly")
                    scaling_diag_matrix = _spd_cholesky(raw_scaling_diag_matrix, dev)
                else:
                    raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                    try:
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                    except Exception:
                        # upstream leaves this second cholesky unguarded; on opt-6.7b+
                        # the absolute 1e-6 shift is too small to make it succeed
                        scaling_diag_matrix = _spd_cholesky(raw_scaling_diag_matrix, dev)
                    if not torch.isfinite(scaling_diag_matrix).all():
                        # cholesky returned WITHOUT raising but with Inf/NaN entries.
                        # CONFIRMED on this build: given a matrix with a single Inf
                        # on the diagonal, torch.linalg.cholesky does not raise and
                        # returns a non-finite factor (eigvalsh raises on the same
                        # input). Unchecked, this is what made SVD-LLM write an all-NaN result
                        # for opt-6.7b while exiting 0 (jobs 942330_0/_1).
                        print("Warning: cholesky returned a non-finite factor; re-deriving")
                        scaling_diag_matrix = _spd_cholesky(raw_scaling_diag_matrix, dev)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            # UPSTREAM LEAK. The hook in THIS function accumulates into
            # `scaling_diag_matrix` (see the hook above), but this cleanup was
            # copy-pasted from profle_svdllm, where the attribute really is called
            # `raw_scaling_diag_matrix`. Assigning None CREATES that unrelated
            # attribute and the del removes it again, so nothing raises -- and not
            # one byte of the actual Gram is released. It stays attached to every
            # Linear of every layer already processed:
            #     opt-30b  (5*7168^2 + 28672^2)*4 = 4.32 GB/layer x 48 = 207 GB
            #     opt-66b  (5*9216^2 + 36864^2)*4 = 7.13 GB/layer x 64 = 457 GB
            # against a 201 GB cgroup. With the fp16 layer written back each
            # iteration the growth is 5.55 GB/layer, so opt-30b dies around layer
            # 36 of 48 -- exactly where job 960798 was OOM-killed (40:55 at
            # ~72 s/layer). Free the attribute the hook actually set.
            scaling_diag_matrix = raw_scaling_diag_matrix = None
            subset[name].scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].scaling_diag_matrix
            torch.cuda.empty_cache()
        # Back to the host in the SOURCE dtype, not fp32 -- otherwise every layer
        # already processed would sit on the host at double size and the fp16 load
        # would have bought nothing. The fp16 -> fp32 -> fp16 round trip is lossless
        # here because profiling only READS the weights, never writes them.
        layers[i] = layer.to("cpu", _src_dtype)
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat
     
 
@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    # The replacement layers below are constructed fresh and therefore land in
    # torch.get_default_dtype() = fp32, regardless of the model's dtype. As
    # whitening progresses the model silently converts from fp16 to fp32 -- for
    # opt-30b at rho=0.60 that is 60 GB growing to 86 GB, which together with the
    # page cache from reading back the spilled factors OOM-killed job 960798.
    # Building them in the model's own dtype keeps the total flat. The factors
    # themselves are already written in the source dtype (see `dtype =
    # subset[name].weight.data.dtype` below) and the SVD still runs in fp32, so
    # this changes only the dtype of the freshly-allocated LayerNorms and biases,
    # which are copied over from the original layer anyway.
    _model_dtype = next(iter(model.parameters())).dtype
    _prev_default = torch.get_default_dtype()
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        #### Replace Attn, MLP ####
        # One guard around the whole block rather than one per constructor, so a
        # family added later cannot silently escape it.
        torch.set_default_dtype(_model_dtype)
        try:
            if "llama" in model_name or "vicuna" in model_name:
                svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio, layer_idx=i)
                svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
            elif "mistral" in model_name:
                svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
                svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
            elif 'opt' in model_name:
                svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        finally:
            torch.set_default_dtype(_prev_default)
        #### Replace Attn, MLP ####
        for name in subset:
            # Capture the SOURCE dtype BEFORE .float(), not after. As written
            # upstream `dtype = W.dtype` is unconditionally torch.float32 --
            # the line above just upcast -- so the factors are always written
            # back fp32, which is exactly why an fp16 model load produced
            # "expected mat1 and mat2 to have the same dtype" against the
            # untouched fp16 layers. Taking it first is a strict no-op for an
            # fp32-loaded model (the only kind every validated run used) and is
            # what makes an fp16 load possible at all: opt-66b in fp32 is 263 GB
            # against a 201 GB cgroup. The SVD still runs in fp32 (line above);
            # only the storage dtype of the factors changes, and run_svdllm.py
            # halves the model before evaluation anyway -- the same single
            # fp32 -> fp16 rounding either way.
            dtype = subset[name].weight.data.dtype
            W = subset[name].weight.data.float().to(dev)
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                # This branch never fires on opt-125m (verified across every
                # opt-125m run: 0 hits), so changing it cannot move the validated
                # 72.0123/356.9292/165.3395 numbers. Upstream adds an ABSOLUTE
                # 1e-6 to the Cholesky FACTOR, whose diagonal reaches ~3e6 here --
                # below one ulp, so the retry raises the identical error. Scale
                # the ridge to the factor, and fall back to a pseudo-inverse.
                _scale = torch.diagonal(scaling_diag_matrix).abs().max().clamp(min=1.0)
                _eye = torch.eye(scaling_diag_matrix.shape[0],
                                 dtype=scaling_diag_matrix.dtype, device=dev)
                # Range extended past 1e-4: the Gram diagonals reach ~1e14 (bounded
                # from ASVD's cached per-channel activation stats: opt-30b's worst
                # channel is mean|x| = 2.08e4, so 524288 * 2.08e4^2 = 2.27e14 --
                # 24 orders below fp32's ceiling, so overflow is NOT the cause).
                # The remaining suspect is a condition number past 1/eps, where
                # pinv's SVD can fail to converge and return NaN. A large ridge
                # gives a less faithful whitening but is still whitening, which
                # beats degrading the matrix to plain SVD.
                scaling_matrix_inv = None
                for _rel in (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
                    _cand = scaling_diag_matrix + (_rel * _scale) * _eye
                    try:
                        _inv = torch.linalg.inv(_cand)
                    except Exception:
                        continue
                    if torch.isfinite(_inv).all():
                        print(f"Info: relative ridge {_rel:g}*{_scale:.3e} restored invertibility")
                        scaling_diag_matrix, scaling_matrix_inv = _cand, _inv
                        break
                if scaling_matrix_inv is None:
                    print("Info: falling back to pinv")
                    try:
                        scaling_matrix_inv = torch.linalg.pinv(scaling_diag_matrix)
                    except Exception as exc:
                        # pinv is itself an SVD and can RAISE "failed to
                        # converge" on these matrices, not merely return
                        # garbage -- observed on opt-350m, whose post-LayerNorm
                        # architecture (do_layer_norm_before=False) leaves the
                        # whitening Grams worse conditioned than any pre-LN OPT.
                        # The existing guard below only caught the non-finite
                        # case, so the raise escaped and killed the run. Fall
                        # through to the counted identity degradation instead.
                        print(f"Info: pinv failed ({type(exc).__name__}); "
                              f"degrading this matrix")
                        scaling_matrix_inv = torch.full_like(
                            scaling_diag_matrix, float("nan"))
                if not torch.isfinite(scaling_matrix_inv).all():
                    # pinv returns finite values for any finite input, so reaching
                    # here means the Cholesky factor itself is poisoned. There is no
                    # meaningful whitening transform to apply: fall back to the
                    # identity, i.e. plain SVD truncation for THIS matrix only.
                    # That is a real change of method, so it is counted and the
                    # count is written into the result JSON -- never silent.
                    print(f"Warning: DEGRADED layer {i} {name} to unwhitened SVD "
                          f"(whitening factor was non-finite)")
                    WHITENING_DEGRADED.append(f"{i}.{name}")
                    scaling_diag_matrix = torch.eye(
                        scaling_diag_matrix.shape[0],
                        dtype=scaling_diag_matrix.dtype, device=dev)
                    scaling_matrix_inv = scaling_diag_matrix.clone()
                del _eye
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio, layer_idx=i)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"truncted error: {self.error}")
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)