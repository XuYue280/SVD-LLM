import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig
from transformers.cache_utils import Cache

# Use the *installed* transformers implementations for everything that is not
# the factored projection itself, so the compressed attention runs exactly the
# same math (RoPE flavour, GQA expansion, mask handling, KV-cache protocol) as
# the dense baseline of whatever transformers version is in the environment.
# The stale local copies further down this file are kept only for backwards
# compatibility with anything that still imports them.
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding as HFLlamaRotaryEmbedding,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SVD_LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ratio=1
    ):
        super().__init__()
        self.ratio = ratio
        low_rank = int(intermediate_size * hidden_size * self.ratio / (intermediate_size + hidden_size))
        self.gate_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        
        self.down_u_proj = nn.Linear(low_rank, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(intermediate_size, low_rank, bias=False)
        
        self.up_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))


def _svd_low_rank(out_features: int, in_features: int, ratio: float) -> int:
    """Rank kept for a weight of shape [out_features, in_features].

    This is *exactly* the formula used by ``SVDLLM.whitening()``:
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
    so the ``nn.Linear`` shapes declared here already match the tensors that
    ``whitening()`` will later slam into ``.weight.data``.  For a square weight
    this reduces to ``int(dim * ratio / 2)``, i.e. the value the previous
    implementation hardcoded, so nothing changes for Llama-1/2 q/k/v/o.
    """
    return int(out_features * in_features * ratio / (out_features + in_features))


class SVD_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper.

    Mirrors ``transformers.models.llama.modeling_llama.LlamaAttention`` /
    ``LlamaSdpaAttention`` of the installed transformers version.  The *only*
    difference is that each of q/k/v/o is applied as a factored pair
    ``<x>_u_proj(<x>_v_proj(h))`` instead of a single ``nn.Linear``.

    Supports GQA/MQA (``num_key_value_heads < num_attention_heads``, e.g.
    Llama-3/3.1) and is bit-identical in structure to the MHA path when
    ``num_key_value_heads == num_attention_heads`` (``repeat_kv`` is then the
    identity).
    """

    def __init__(self, config: LlamaConfig, ratio=1, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        _head_dim = getattr(config, "head_dim", None)
        self.head_dim = _head_dim if _head_dim is not None else self.hidden_size // self.num_heads
        # Llama-1/2 have num_key_value_heads == num_attention_heads (plain MHA);
        # Llama-3/3.1 use GQA (8 kv heads vs 32 q heads).
        self.num_key_value_heads = getattr(config, "num_key_value_heads", None) or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.is_causal = True
        self.ratio = ratio  # 1 means no truncate, just keep normal attn

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by num_key_value_heads "
                f"({self.num_key_value_heads})."
            )

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        o_in = self.num_heads * self.head_dim

        # Ranks are per-matrix: for Llama-3.1-8B @ ratio=0.6 this gives
        # q: 1228, k: 491, v: 491, o: 1228.  `whitening()` overwrites
        # `.weight.data` anyway, but declaring the right shapes keeps the module
        # usable (and correctly sized) on its own.
        q_rank = _svd_low_rank(q_out, self.hidden_size, self.ratio)
        kv_rank = _svd_low_rank(kv_out, self.hidden_size, self.ratio)
        o_rank = _svd_low_rank(self.hidden_size, o_in, self.ratio)

        # Llama has attention_bias=False; `whitening()` never assigns attention
        # biases for llama/mistral, so the factors stay bias-free.
        self.q_u_proj = nn.Linear(q_rank, q_out, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, q_rank, bias=False)

        self.k_u_proj = nn.Linear(kv_rank, kv_out, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, kv_rank, bias=False)

        self.v_u_proj = nn.Linear(kv_rank, kv_out, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, kv_rank, bias=False)

        self.o_u_proj = nn.Linear(o_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(o_in, o_rank, bias=False)

        # Config-aware ctor: picks up rope_theta (500000 for Llama-3.1) *and*
        # rope_scaling ({"rope_type": "llama3", ...}).  The old positional API
        # silently hardcoded base=10000 and no scaling.
        self.rotary_emb = HFLlamaRotaryEmbedding(config=self.config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # ---- the one and only deviation from stock: factored projections ----
        query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        value_states = self.v_u_proj(self.v_v_proj(hidden_states))
        # ---------------------------------------------------------------------

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = hf_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # GQA: expand kv heads to match q heads. No-op when num_key_value_groups == 1.
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        use_sdpa = getattr(self.config, "_attn_implementation", "eager") == "sdpa" and not output_attentions

        if use_sdpa:
            # Mirrors LlamaSdpaAttention.forward
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
            # custom attn_mask, Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)
            attn_output = self.o_u_proj(self.o_v_proj(attn_output))
            return attn_output, None, past_key_value

        # Mirrors LlamaAttention.forward (eager)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
