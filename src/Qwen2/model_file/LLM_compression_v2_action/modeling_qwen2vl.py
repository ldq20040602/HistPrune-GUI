# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2-VL model."""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None

logger = logging.get_logger(__name__)


def find_indices(tensor, value=151646):
    tensor = tensor.flatten()  # Ensure the tensor is one-dimensional
    matches = (tensor == value)

    # Roll the matches tensor by 1
    rolled = torch.roll(matches, shifts=1)
    rolled[0] = False  # Set the first element to False to prevent false positives

    starts = (matches & ~rolled).nonzero(as_tuple=True)[0]
    ends = (~matches & rolled).nonzero(as_tuple=True)[0]

    return list(zip(starts.tolist(), ends.tolist()))


_CONFIG_FOR_DOC = "Qwen2VLConfig"


def find_indices_in_batch(tensor, value=151646):
    batch_size = tensor.shape[0]
    result = []
    for i in range(batch_size):
        result.append(find_indices(tensor[i], value))
    return result


def find_OAS_segments(tensor, image_token_id: int):
    """
    基于 Qwen2-VL 的图像占位符（config.image_token_id）在 input_ids 中检索所有“连续的图像 token 片段”，
    并返回这些片段在序列维度上的区间 [start, end)（左闭右开）。
    注意：不再依赖旧版的硬编码哨兵（1906/198/16448），仅以连续的 image_token_id 作为图像段判定依据。
    """
    tensor = tensor.flatten()
    pos = (tensor == image_token_id).nonzero(as_tuple=True)[0]
    segments = []
    if pos.numel() == 0:
        return segments
    # 将等于 image_token_id 的位置合并为若干连续区间
    start = pos[0].item()
    prev = pos[0].item()
    for idx in pos[1:]:
        idx = idx.item()
        if idx == prev + 1:
            prev = idx
        else:
            segments.append([start, prev + 1])
            start = idx
            prev = idx
    segments.append([start, prev + 1])
    return segments


def Find_OAS(tensor, image_token_id: int):
    batch_size = tensor.shape[0]
    result = []
    for i in range(batch_size):
        result.append(find_OAS_segments(tensor[i], image_token_id))
    return result


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class Qwen2VLRotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config: Optional[Qwen2VLConfig] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size: int = 14,
            temporal_patch_size: int = 2,
            in_channels: int = 3,
            embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
            self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class VisionFlashAttention2(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
            self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output


class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
            self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


QWEN2_VL_VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
}


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = QWEN2_VL_VISION_ATTENTION_CLASSES[attn_implementation](
            config.embed_dim, num_heads=config.num_heads
        )
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        min_dtype: float,
        cache_position: torch.Tensor,
        batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1

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
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

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

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2VL flash attention module, following Qwen2VL attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
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

        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                    getattr(self.config, "sliding_window", None) is not None
                    and kv_seq_len > self.config.sliding_window
                    and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
                self.config.use_sliding_window
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLSdpaAttention(Qwen2VLAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
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
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
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
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_VL_ATTENTION_CLASSES = {
    "eager": Qwen2VLAttention,
    "flash_attention_2": Qwen2VLFlashAttention2,
    "sdpa": Qwen2VLSdpaAttention,
}


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2VisionTransformerPretrainedModel(Qwen2VLPreTrainedModel):
    config_class = Qwen2VLVisionConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        # FastV pruning flags (default disabled)
        # - enable_fastv_pruning: 是否启用基于注意力的历史截图稀疏剪枝（步骤0占位，默认关闭）
        # - fastv_segment_keep_ratios: 分段保留比例（最近->更远）
        # - _fastv_last_attn_weights: 在第 drop_k-1 层收集到的注意力权重缓存（[B, H, Q, K]）
        self.enable_fastv_pruning: bool = False
        self.fastv_segment_keep_ratios: List[float] = []
        self._fastv_last_attn_weights = None
        self._fastv_keep_mask = None

        # PDrop pruning (single text-token query), default disabled
        self.enable_pdrop_pruning: bool = False
        # per-history-image keep ratios (recent -> older)
        self.pdrop_keep_ratios: List[float] = []
        # backward compatibility: old single keep ratio
        self.pdrop_keep_ratio_m: float = 0.5
        self._pdrop_keep_mask = None  # [B, L] bool
        self._pdrop_query_index: Optional[int] = None

        # UIgraph pruning (region-quota guided), default disabled
        self.enable_uigraph_pruning: bool = False
        self._uigraph_token_regions = None  # [B, L] int64, -1 for non-history positions
        self._uigraph_keep_mask = None  # [B, L] bool
        self._uigraph_info = None  # list per batch: each is list per history image with dict{'patch_assign': list[int]}

        # Sobel pruning (edge-only), default disabled
        self.sobel_enable: bool = False
        self._sobel_keep_mask = None  # [B, L] bool
        self._sobel_info = None  # list per batch: each is list per history image with dict{'is_edge': list[bool]}

        # Sparse Greedy pruning (with Sobel prior), default disabled
        self.enable_DivPrune_pruning: bool = False
        self.DivPrune_ratios: List[float] = []  # per-history-image keep ratios (recent -> older).
        self._DivPrune_keep_mask = None  # [B, L] bool
        self._DivPrune_info = None  # list per batch: each is list per history image with dict{'is_edge': list[bool]}

        # Random pruning (segment-wise per history image), default disabled
        self.enable_random_pruning: bool = False
        self.random_keep_ratios: List[float] = []  # per-history-image keep ratios (recent -> older)
        # backward compatibility
        self.random_prune_ratios: List[float] = []
        self._random_keep_mask = None  # [B, L] bool

        # DART pruning (diversity-aware, per-history-image), default disabled
        self.enable_dart_pruning: bool = False
        self.dart_keep_ratios: List[float] = []  # per-history-image keep ratios (recent -> older)
        self.dart_num_pivots: int = 10
        self.dart_debug: bool = False
        self._dart_keep_mask = None  # [B, L] bool

        # SparseVLM pruning (Stage1 raters + Stage2 global visual pruning), default disabled
        self.enable_sparsevlm_pruning: bool = False
        self.sparsevlm_keep_ratio: float = 0.4
        self.sparsevlm_tau: float = 0.3
        self.sparsevlm_cluster_ratio: float = 0.03
        self._sparsevlm_keep_mask = None  # [B, L] bool
        self._sparsevlm_reconstructed_info = None  # List of dicts for each sample in batch

        # FLOPs hook: record sequence length after pruning at drop_k (prefill stage only)
        # - Mind2web_eval.py will read this value as n_after
        self._n_after_hook: Optional[int] = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 函数作用：依据 OAS_index 指定的区间（图像占位与对应动作文本的段）从隐藏状态序列中删除这些 token，
    # 并同步裁剪两组位置嵌入(position_embeddings[0/1])与重新构建批对齐后的注意力掩码，
    # 以便在后续层以更短的序列长度进行计算，实现加速。
    # 参数：
    # - hidden: 原始隐藏状态张量，形如 [batch, seq_len, hidden_dim]
    # - OAS_index: 每个样本对应的段索引列表，例如 [[O_start, O_end, A_start, A_end], ...]，按样本组织
    # - input_ids: 原始输入 token，用于统计各样本非 pad 的有效长度与计算对齐后的 pad 长度
    # - position_embeddings: 两组位置嵌入列表 [pos0, pos1]，每组形如 [1, batch, seq_len, dim]
    def Drop_hiddenstates(self, hidden, OAS_index, input_ids, position_embeddings):
        """
        基础裁剪函数（用于 all-False 模式）：
        仅删除历史截图的 image tokens，保留所有动作文本、当前截图及其后的指令。
        """
        B, L, H = hidden.shape
        device = hidden.device
        # 1. 构造一个全 True 的掩码，表示默认全部保留
        keep_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            idx_list = OAS_index[b] if OAS_index is not None else []
            # 如果没有图像段或只有一段（即只有当前图），则不进行任何删除
            if not isinstance(idx_list, list) or len(idx_list) <= 1:
                continue
            
            # 2. 识别历史截图段：除了最后一段（当前图），前面的都是历史
            # 约定 idx_list 每一项是 [start, end]
            hist_segments = idx_list[:-1]
            
            for seg in hist_segments:
                if isinstance(seg, list) and len(seg) >= 2:
                    s, e = int(seg[0]), int(seg[1])
                    if e > s and s >= 0 and e <= L:
                        # 将历史截图区间在 mask 中设为 False (删除)
                        keep_mask[b, s:e] = False

        # 3. 直接调用基于 Mask 的统一裁剪逻辑
        # 这会自动处理 hidden_states 提取、位置编码同步对齐以及注意力掩码重建
        return self.Drop_hiddenstates_fastv(hidden, keep_mask, input_ids, position_embeddings)

    # UIgraph 剪枝版：与 FastV 相同的按掩码裁剪逻辑，单独保留接口便于模式区分
    def Drop_hiddenstates_UIgraph(self, hidden, keep_mask, input_ids, position_embeddings):
        return self.Drop_hiddenstates_fastv(hidden, keep_mask, input_ids, position_embeddings)

    # FastV 剪枝版：基于 keep_mask（True=保留）按序列维度裁剪，再做批内对齐与注意力掩码重建
    # - hidden: [B, L, H]
    # - keep_mask: [B, L] (bool)
    # - input_ids: [B, L]
    # - position_embeddings: [cos, sin] 各为 [1, B, L, D]
    # 返回：(padded_sequence[B,L',H], [pe0[1,B,L',D], pe1[1,B,L',D]], padded_attention_mask[B,L'])
    def Drop_hiddenstates_fastv(self, hidden, keep_mask, input_ids, position_embeddings):
        hidden_states = hidden
        B, L, H = hidden_states.shape
        device = hidden_states.device
        # 原有效长度（按 pad id 151643 判定，与旧实现保持一致）
        seq_length_before = [(input_ids[b] != 151643).sum().item() for b in range(B)]
        kept_lens = [int(keep_mask[b].sum().item()) for b in range(B)]
        pad_after = max(kept_lens) if B > 0 else 0
        # 注意力掩码：前 kept_len 为 True
        arange_tensor = torch.arange(pad_after, device=device).unsqueeze(0).expand(B, -1)
        padded_attention_mask = (arange_tensor < torch.tensor(kept_lens, device=device).unsqueeze(1))

        padded_sequence = []
        padded_position_embedding_0 = []
        padded_position_embedding_1 = []

        # 选择一个 pad_hidden（如原实现：若批内存在原始padding的样本则用其末尾token，否则用第一个样本末尾token）
        pad_hidden = None
        for b in range(B):
            if seq_length_before[b] < L:
                pad_hidden = hidden_states[b:b + 1, -1:, :].squeeze(0)  # [1,H]
                break
        if pad_hidden is None and B > 0:
            pad_hidden = hidden_states[0:1, -1:, :].squeeze(0)

        for b in range(B):
            idx_keep = torch.nonzero(keep_mask[b], as_tuple=False).squeeze(-1)
            cur_seq = hidden_states[b, idx_keep, :] if idx_keep.numel() > 0 else hidden_states.new_zeros((0, H))
            # 位置嵌入沿 seq 维（dim=2）索引
            cur_pos0 = position_embeddings[0][:, b:b + 1, idx_keep, :] if idx_keep.numel() > 0 else position_embeddings[
                                                                                                        0][:, b:b + 1,
                                                                                                    0:0, :]
            cur_pos1 = position_embeddings[1][:, b:b + 1, idx_keep, :] if idx_keep.numel() > 0 else position_embeddings[
                                                                                                        1][:, b:b + 1,
                                                                                                    0:0, :]

            # 尾部 pad 到 pad_after
            if cur_seq.shape[0] < pad_after:
                need = pad_after - cur_seq.shape[0]
                if pad_hidden is None:
                    pad_hidden = hidden_states[0:1, -1:, :].squeeze(0)
                cur_pad = pad_hidden.unsqueeze(0).expand(need, -1)  # [need,H]
                cur_seq = torch.cat([cur_seq, cur_pad], dim=0)

                pos0_tail = position_embeddings[0][:, b:b + 1, -1:, :].expand(-1, -1, need, -1)
                pos1_tail = position_embeddings[1][:, b:b + 1, -1:, :].expand(-1, -1, need, -1)
                cur_pos0 = torch.cat([cur_pos0, pos0_tail], dim=2)
                cur_pos1 = torch.cat([cur_pos1, pos1_tail], dim=2)
            elif cur_seq.shape[0] > pad_after:
                cur_seq = cur_seq[:pad_after]
                cur_pos0 = cur_pos0[:, :, :pad_after, :]
                cur_pos1 = cur_pos1[:, :, :pad_after, :]

            padded_sequence.append(cur_seq)
            padded_position_embedding_0.append(cur_pos0)
            padded_position_embedding_1.append(cur_pos1)

        padded_sequence = torch.stack(padded_sequence, dim=0)
        padded_position_embedding_0 = torch.cat(padded_position_embedding_0, dim=1)
        padded_position_embedding_1 = torch.cat(padded_position_embedding_1, dim=1)

        return padded_sequence, [padded_position_embedding_0, padded_position_embedding_1], padded_attention_mask

    # 仅用于 FastV（步骤1）：在第 drop_k-1 层收集注意力分布。
    # 注意：当底层实现为 flash_attention_2 时，该层不会返回注意力；
    #       因此这里提供一个“手工计算”注意力概率的方法，复用该层的权重，避免改动层类型。
    def _fastv_compute_attn_probs(self, decoder_layer, hidden_states, attention_mask, position_ids, cache_position,
                                  position_embeddings):
        # 使用与层内一致的前置层归一化
        hs = decoder_layer.input_layernorm(hidden_states)
        attn_mod = decoder_layer.self_attn
        bsz, q_len, _ = hs.size()
        # 线性投影
        query_states = attn_mod.q_proj(hs)
        key_states = attn_mod.k_proj(hs)
        # 形状变换
        query_states = query_states.view(bsz, q_len, attn_mod.num_heads, attn_mod.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn_mod.num_key_value_heads, attn_mod.head_dim).transpose(1, 2)
        # 位置编码（RoPE）
        if position_embeddings is None:
            cos, sin = attn_mod.rotary_emb(key_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, attn_mod.rope_scaling["mrope_section"]
        )
        # 重复 kv 头到与 num_heads 对齐
        key_states = repeat_kv(key_states, attn_mod.num_key_value_groups)
        # 注意力 logits
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_mod.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        return attn_probs

    # 仅用于 FastV（步骤2）：基于 OAS_index 生成历史截图候选集合的 mask，排除当前图像与 pad。
    def _fastv_build_candidate_mask(self, OAS_index, input_ids, pad_token_id: int = 151643):
        # input_ids: [B, L]
        B, L = input_ids.shape
        device = input_ids.device
        candidate_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            cur_index = OAS_index[b] if OAS_index is not None else []
            if len(cur_index) == 0:
                continue
            # 无历史的特殊记号：首元素 []
            if isinstance(cur_index[0], list) and len(cur_index[0]) == 0:
                continue
            # 历史段：除去“当前图像段”（位于末尾）
            last_hist_j = len(cur_index) - 2
            for j, seg in enumerate(cur_index):
                if not isinstance(seg, list) or len(seg) == 0:
                    continue
                if j <= last_hist_j:
                    s, e = int(seg[0]), int(seg[1])
                    s = max(0, min(s, L))
                    e = max(0, min(e, L))
                    if e > s:
                        candidate_mask[b, s:e] = True
        # 过滤 pad
        valid_mask = (input_ids != pad_token_id)
        candidate_mask &= valid_mask
        return candidate_mask

    # 仅用于 FastV（步骤3）：聚合注意力分数并生成 keep_mask（True=保留）
    def _fastv_build_keep_mask(self, attn_probs: torch.Tensor, candidate_mask: torch.Tensor, m: float):
        # attn_probs: [B, H, Q, K]
        # candidate_mask: [B, K] (bool)
        B, H, Q, K = attn_probs.shape
        device = attn_probs.device
        keep_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        # 聚合：对每个 key 被所有 query 和 head 的关注度求和
        # 使用基于 Prompt (P1+P2) 的 Query 掩码来聚合注意力；若无掩码则退回全Q聚合
        B, H, Q, K = attn_probs.shape
        q_ranges = getattr(self, '_prompt_query_ranges', None)
        if q_ranges is not None:
            qmask = torch.zeros(B, Q, dtype=attn_probs.dtype, device=attn_probs.device)
            for b in range(B):
                item = q_ranges[b] if isinstance(q_ranges, list) and b < len(q_ranges) else q_ranges
                if isinstance(item, dict):
                    for key in ['p1', 'p2']:
                        if key in item and isinstance(item[key], (list, tuple)) and len(item[key]) == 2:
                            s, e = int(item[key][0]), int(item[key][1])
                            s = max(0, min(s, Q));
                            e = max(0, min(e, Q))
                            if e > s:
                                qmask[b, s:e] = 1.0
            scores = (attn_probs * qmask.view(B, 1, Q, 1)).sum(dim=(1, 2))  # [B, K]
        else:
            scores = attn_probs.sum(dim=(1, 2))  # [B, K]
        for b in range(B):
            cand_idx = torch.nonzero(candidate_mask[b], as_tuple=False).squeeze(-1)
            N = cand_idx.numel()
            if N == 0:
                continue
            drop_b = int((m * N) // 1)  # floor
            if drop_b <= 0:
                continue
            # 选择候选集内得分最低的 drop_b 个位置
            cand_scores = scores[b, cand_idx]
            # 若 drop_b >= N，全部候选都删除
            if drop_b >= N:
                keep_mask[b, cand_idx] = False
                continue
            vals, idx = torch.topk(cand_scores, k=drop_b, largest=False)
            drop_indices = cand_idx[idx]
            keep_mask[b, drop_indices] = False
        return keep_mask

    # 分段 FastV：按历史截图远近为每张历史图单独在段内删除比例（ratios：最近->第4近）
    def _fastv_build_keep_mask_per_image(self, attn_probs: torch.Tensor, OAS_index, input_ids, ratios,
                                         pad_token_id: int = 151643):
        # attn_probs: [B,H,Q,K]
        B, H, Q, K = attn_probs.shape
        device = attn_probs.device
        keep_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        # 聚合注意力（支持 Prompt Query 掩码）
        q_ranges = getattr(self, '_prompt_query_ranges', None)
        if q_ranges is not None:
            qmask = torch.zeros(B, Q, dtype=attn_probs.dtype, device=device)
            for b in range(B):
                item = q_ranges[b] if isinstance(q_ranges, list) and b < len(q_ranges) else q_ranges
                if isinstance(item, dict):
                    for key in ['p1', 'p2']:
                        if key in item and isinstance(item[key], (list, tuple)) and len(item[key]) == 2:
                            s, e = int(item[key][0]), int(item[key][1])
                            s = max(0, min(s, Q));
                            e = max(0, min(e, Q))
                            if e > s:
                                qmask[b, s:e] = 1.0
            scores = (attn_probs * qmask.view(B, 1, Q, 1)).sum(dim=(1, 2))
        else:
            scores = attn_probs.sum(dim=(1, 2))
        valid_all = (input_ids != pad_token_id)
        for b in range(B):
            idx_list = OAS_index[b] if OAS_index is not None else []
            if not isinstance(idx_list, list) or len(idx_list) <= 1:
                continue
            hist_list = idx_list[:-1]
            k = min(4, len(hist_list))
            if k <= 0:
                continue
            sub = hist_list[-k:]
            for i in range(k):  # i=0 最近, i=1 第二近 ...
                seg = sub[-1 - i]
                if not isinstance(seg, list) or len(seg) < 2:
                    continue
                s, e = int(seg[0]), int(seg[1])
                s = max(0, min(s, K));
                e = max(0, min(e, K))
                if e <= s:
                    continue
                # 选择比例
                if isinstance(ratios, (list, tuple)) and len(ratios) > 0:
                    keep_r = ratios[i] if i < len(ratios) else ratios[-1]
                else:
                    keep_r = 1.0
                keep_r = 0.0 if keep_r < 0.0 else (1.0 if keep_r > 1.0 else keep_r)
                # 段有效位置
                seg_valid = valid_all[b, s:e]
                if not torch.any(seg_valid):
                    continue
                abs_range = torch.arange(s, e, device=device)
                abs_idx = abs_range[seg_valid]
                N = abs_idx.numel()
                if N <= 0:
                    continue
                drop_n = int(((1.0 - keep_r) * N) // 1)
                if drop_n <= 0:
                    continue
                seg_scores = scores[b, abs_idx]
                if drop_n >= N:
                    keep_mask[b, abs_idx] = False
                    continue
                vals, order = torch.topk(seg_scores, k=drop_n, largest=False)
                drop_indices = abs_idx[order]
                keep_mask[b, drop_indices] = False
        return keep_mask

    # PDrop: 仅使用文本最后一个 token 作为 query，按注意力删除历史截图低分 token。
    # ratios 语义与 FastV 对齐：最近->更远 的分段 keep ratios。
    def _pdrop_build_keep_mask(self, attn_probs: torch.Tensor, OAS_index, input_ids, ratios,
                               query_index: Optional[int] = None, pad_token_id: int = 151643):
        B, H, Q, K = attn_probs.shape
        device = attn_probs.device
        keep_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        valid_all = (input_ids != pad_token_id)

        if query_index is None:
            query_index = int(Q - 1)
        qidx = int(max(0, min(int(query_index), Q - 1)))

        # 聚合该单 query 在各 head 上对 key 的注意力
        scores = attn_probs[:, :, qidx, :].sum(dim=1)  # [B, K]

        for b in range(B):
            idx_list = OAS_index[b] if OAS_index is not None else []
            if not isinstance(idx_list, list) or len(idx_list) <= 1:
                continue

            hist_list = idx_list[:-1]
            k = min(4, len(hist_list))
            if k <= 0:
                continue
            sub = hist_list[-k:]

            # i=0 最近, i=1 第二近 ...
            for i in range(k):
                seg = sub[-1 - i]
                if not isinstance(seg, list) or len(seg) < 2:
                    continue
                s, e = int(seg[0]), int(seg[1])
                s = max(0, min(s, K))
                e = max(0, min(e, K))
                if e <= s:
                    continue

                if isinstance(ratios, (list, tuple)) and len(ratios) > 0:
                    keep_r = float(ratios[i] if i < len(ratios) else ratios[-1])
                else:
                    keep_r = 1.0
                keep_r = 0.0 if keep_r < 0.0 else (1.0 if keep_r > 1.0 else keep_r)

                seg_valid = valid_all[b, s:e]
                if not torch.any(seg_valid):
                    continue

                abs_range = torch.arange(s, e, device=device)
                abs_idx = abs_range[seg_valid]
                N = abs_idx.numel()
                if N <= 0:
                    continue

                drop_n = int(((1.0 - keep_r) * N) // 1)
                if drop_n <= 0:
                    continue
                if drop_n >= N:
                    keep_mask[b, abs_idx] = False
                    continue

                seg_scores = scores[b, abs_idx]
                _, order = torch.topk(seg_scores, k=drop_n, largest=False)
                drop_indices = abs_idx[order]
                keep_mask[b, drop_indices] = False

        return keep_mask

    # 仅用于 UIgraph：基于区域配额与注意力构造 keep_mask（True=保留）
    # uigraph_info: list 长度为 batch，每个元素是该样本历史图像的列表；
    #   每个历史图像项应包含 {'patch_assign': List[int], 'area_ratio': List[float]}。
    def _uigraph_build_keep_mask(self, attn_probs: torch.Tensor, OAS_index, uigraph_info, input_ids):
        B, H, Q, K = attn_probs.shape
        device = attn_probs.device
        keep_mask = torch.ones(B, K, dtype=torch.bool, device=device)
        # 使用基于 Prompt (P1+P2) 的 Query 掩码来聚合注意力；若无掩码则退回全Q聚合
        B, H, Q, K = attn_probs.shape
        q_ranges = getattr(self, '_prompt_query_ranges', None)
        if q_ranges is not None:
            qmask = torch.zeros(B, Q, dtype=attn_probs.dtype, device=attn_probs.device)
            for b in range(B):
                item = q_ranges[b] if isinstance(q_ranges, list) and b < len(q_ranges) else q_ranges
                if isinstance(item, dict):
                    for key in ['p1', 'p2']:
                        if key in item and isinstance(item[key], (list, tuple)) and len(item[key]) == 2:
                            s, e = int(item[key][0]), int(item[key][1])
                            s = max(0, min(s, Q));
                            e = max(0, min(e, Q))
                            if e > s:
                                qmask[b, s:e] = 1.0
            scores = (attn_probs * qmask.view(B, 1, Q, 1)).sum(dim=(1, 2))  # [B, K]
        else:
            scores = attn_probs.sum(dim=(1, 2))  # [B, K]
        pad_token_id = 151643
        valid_mask = (input_ids != pad_token_id)
        for b in range(B):
            # 样本级信息
            idx_list = OAS_index[b] if OAS_index is not None else []
            if (uigraph_info is None) or (b >= len(uigraph_info)):
                continue
            img_infos = uigraph_info[b]
            if not isinstance(idx_list, list) or len(idx_list) == 0 or len(img_infos) == 0:
                continue
            # 标准化 OAS 段为二元区间 [s, e)，以避免四元组被后续逻辑直接跳过
            proj_idx_list = []
            for _seg in idx_list:
                if isinstance(_seg, list) and len(_seg) >= 2:
                    proj_idx_list.append([int(_seg[0]), int(_seg[1])])
                elif isinstance(_seg, list) and len(_seg) == 0:
                    proj_idx_list.append([])
                else:
                    # 占位以保持与 img_infos 的对齐关系
                    proj_idx_list.append([])
            idx_list = proj_idx_list
            # 历史图像数量（排除最后的当前图像）
            num_hist = max(0, len(idx_list) - 1)
            num_hist = min(num_hist, len(img_infos))
            for hidx in range(num_hist):
                seg = idx_list[hidx]
                if not isinstance(seg, list) or len(seg) != 2:
                    continue
                s, e = int(seg[0]), int(seg[1])
                if e <= s or s < 0 or e > K:
                    continue
                # 限定在有效（非pad）范围
                seg_valid = valid_mask[b, s:e]
                if not torch.any(seg_valid):
                    continue
                info = img_infos[hidx]
                if (info is None) or ('patch_assign' not in info) or ('area_ratio' not in info):
                    continue
                patch_assign = info['patch_assign']
                # 转成张量
                patch_assign = torch.as_tensor(patch_assign, device=device, dtype=torch.long)
                # 如果长度与段长不一致则跳过（保守）
                if patch_assign.numel() != (e - s):
                    continue
                # 区域枚举
                max_region_id = int(patch_assign.max().item()) if patch_assign.numel() > 0 else -1
                # 逐区域计算配额并在区域内按注意力删除
                for rid in range(max_region_id + 1):
                    # 该区域的面积占比
                    area_ratio_list = info['area_ratio']
                    p_area = 0.0
                    if rid < len(area_ratio_list):
                        p_area = float(area_ratio_list[rid])
                    # 分段函数：<0.1 -> 0; 0.1~0.5 -> 0.3; >0.5 -> 0.8
                    if p_area < 0.1:
                        prune_ratio = 0.0
                    elif p_area <= 0.5:
                        prune_ratio = 0.3
                    else:
                        prune_ratio = 0.8
                    # 该区域内 token 索引（相对段内）
                    rel_idx = torch.nonzero(patch_assign == rid, as_tuple=False).squeeze(-1)
                    if rel_idx.numel() == 0:
                        continue
                    abs_idx = s + rel_idx
                    # 仅保留有效位置
                    abs_idx = abs_idx[seg_valid[rel_idx]]
                    N = abs_idx.numel()
                    if N == 0:
                        continue
                    drop_r = int((prune_ratio * N) // 1)
                    if drop_r <= 0:
                        continue
                    # 在区域内按注意力从低到高删除
                    region_scores = scores[b, abs_idx]
                    if drop_r >= N:
                        keep_mask[b, abs_idx] = False
                        continue
                    vals, order = torch.topk(region_scores, k=drop_r, largest=False)
                    drop_indices = abs_idx[order]
                    keep_mask[b, drop_indices] = False
        return keep_mask

    # Sobel edge-only: 基于 Sobel 的 is_edge 列表生成 keep_mask（True=保留），不依赖注意力。
    # sobel_info: list per batch: each is list per history image with dict{'is_edge': list[bool]}（与 Sparse Greedy 相同 schema）
    def _sobel_build_keep_mask(self, OAS_index, sobel_info, input_ids, pad_token_id: int = 151643):
        # input_ids: [B, L]
        B, L = input_ids.shape
        device = input_ids.device
        keep_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        if sobel_info is None or not isinstance(sobel_info, list) or len(sobel_info) < B:
            raise ValueError("Sobel edge-only: sobel_info is missing or invalid")

        for b in range(B):
            idx_list = OAS_index[b] if OAS_index is not None else []
            if not isinstance(idx_list, list) or len(idx_list) <= 1:
                continue
            hist_segments = []
            for seg in idx_list[:-1]:
                if isinstance(seg, list) and len(seg) >= 2:
                    s, e = int(seg[0]), int(seg[1])
                    if e > s and s >= 0 and e <= L:
                        hist_segments.append((s, e))

            info_list = sobel_info[b]
            if not isinstance(info_list, list) or len(info_list) < len(hist_segments):
                raise ValueError("Sobel edge-only: sobel_info list length mismatch with history segments")

            for img_id, (s, e) in enumerate(hist_segments):
                seg_len = int(e - s)
                info = info_list[img_id]
                if not isinstance(info, dict) or 'is_edge' not in info:
                    raise ValueError("Sobel edge-only: missing is_edge in sobel_info")
                is_edge = info.get('is_edge', None)
                if not isinstance(is_edge, list) or len(is_edge) != seg_len:
                    raise ValueError(
                        f"Sobel edge-only is_edge length mismatch: seg_len={seg_len}, is_edge_len={(len(is_edge) if isinstance(is_edge, list) else None)}")

                # 对该历史图段：无边缘的 token 全部删除（但 pad 位置保持为 True 以避免被当成有效 token）
                is_edge_t = torch.as_tensor(is_edge, device=device, dtype=torch.bool)
                seg_ids = input_ids[b, s:e]
                seg_valid = (seg_ids != pad_token_id)
                # drop = valid & ~is_edge
                drop_mask = seg_valid & (~is_edge_t)
                if torch.any(drop_mask):
                    abs_idx = torch.arange(s, e, device=device)[drop_mask]
                    keep_mask[b, abs_idx] = False

        return keep_mask

    # Random pruning: 对历史截图段内的 token 做可复现的随机删除。
    # - OAS_index: Find_OAS 得到的各图像段 [s,e)，最后一段是当前图像；仅对历史段生效。
    # - random_keep_ratios: 最近->更远 的保留比例列表；历史段超过该长度时用最后一个比例；不足则只用已有段。
    # - 仅对非 pad token 生效，pad/非历史位置保持 True。
    def _random_build_keep_mask(self, OAS_index, input_ids, pad_token_id: int = 151643):
        B, L = input_ids.shape
        device = input_ids.device
        keep_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # ratios: recent -> older
        ratios = getattr(self, 'random_keep_ratios', None)
        if not isinstance(ratios, (list, tuple)) or len(ratios) == 0:
            # backward compatibility: old prune-ratio semantics
            old_ratios = getattr(self, 'random_prune_ratios', None)
            if isinstance(old_ratios, (list, tuple)) and len(old_ratios) > 0:
                ratios = [1.0 - float(x) for x in old_ratios]
            else:
                return keep_mask

        # 取一个固定 seed，保证严格可复现
        seed = int(getattr(self, 'random_prune_seed', 0))

        for b in range(B):
            idx_list = OAS_index[b] if OAS_index is not None else []
            if not isinstance(idx_list, list) or len(idx_list) <= 1:
                continue
            # 历史段（按旧->新），排除最后的当前图
            hist_segments = []
            for seg in idx_list[:-1]:
                if isinstance(seg, list) and len(seg) >= 2:
                    s, e = int(seg[0]), int(seg[1])
                    if e > s and s >= 0 and e <= L:
                        hist_segments.append((s, e))
            if len(hist_segments) == 0:
                continue

            # 仅对最近最多4张历史图做分段随机剪枝
            k = min(4, len(hist_segments))
            segs = hist_segments[-k:]  # 旧->新（取最近k张）

            # 转为 recent->older 的 ratio 对齐：segs_recent[0] 是最近一张
            segs_recent = list(reversed(segs))

            for i, (s, e) in enumerate(segs_recent):
                keep_r = float(ratios[i] if i < len(ratios) else ratios[-1])
                keep_r = 0.0 if keep_r < 0.0 else (1.0 if keep_r > 1.0 else keep_r)

                seg_ids = input_ids[b, s:e]
                seg_valid = (seg_ids != pad_token_id)
                if not torch.any(seg_valid):
                    continue
                abs_range = torch.arange(s, e, device=device)
                abs_idx = abs_range[seg_valid]
                N = int(abs_idx.numel())
                if N <= 0:
                    continue

                drop_n = int(((1.0 - keep_r) * N) // 1)
                if drop_n <= 0:
                    continue
                if drop_n >= N:
                    keep_mask[b, abs_idx] = False
                    continue

                # 每个 (sample b, history rank i, segment span s-e) 用独立 generator，保证跨运行一致
                g = torch.Generator(device=device)
                # 用 segment 边界与 b/i 做扰动，避免不同段拿到同一随机序列
                g.manual_seed(seed + (b + 1) * 1000003 + (i + 1) * 9176 + s * 97 + e * 131)

                perm = torch.randperm(N, generator=g, device=device)
                drop_sel = perm[:drop_n]
                drop_indices = abs_idx[drop_sel]
                keep_mask[b, drop_indices] = False

        return keep_mask

    # DART pruning: per-history-image diversity-aware keep mask.
    # - For each history image segment, pick pivots as top-L2 tokens (num_pivots).
    # - For each token, compute score = max cosine similarity to any pivot.
    # - Keep tokens with smallest scores (farthest from pivots) according to keep_ratio.
    # - Only operates within each history image segment; does not cross images.
    def _dart_build_keep_mask(self, hidden_states: torch.Tensor, OAS_index, input_ids, pad_token_id: int = 151643):
        B, L, H = hidden_states.shape
        device = hidden_states.device
        keep_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        dart_debug = bool(getattr(self, 'dart_debug', False))

        ratios = getattr(self, 'dart_keep_ratios', None)
        if not isinstance(ratios, (list, tuple)) or len(ratios) == 0:
            ratios = [1.0]

        num_pivots = int(getattr(self, 'dart_num_pivots', 10))
        if dart_debug:
            try:
                logger.warning_once(
                    f"[DART-DEBUG][build] B={B} L={L} H={H} num_pivots={num_pivots} ratios={list(ratios)} has_oas={OAS_index is not None}"
                )
            except Exception:
                pass
        if num_pivots <= 0:
            if dart_debug:
                logger.warning_once("[DART-DEBUG][build] num_pivots<=0, skip DART keep-mask build")
            return keep_mask

        for b in range(B):
            idx_list = OAS_index[b] if OAS_index is not None else []
            if not isinstance(idx_list, list) or len(idx_list) <= 1:
                continue

            hist_list = idx_list[:-1]  # old -> new
            if len(hist_list) == 0:
                continue

            # Only handle up to 4 most recent history images, align with ratios (recent -> older)
            k = min(4, len(hist_list))
            hist_sub = hist_list[-k:]
            # recent -> older
            hist_recent = list(reversed(hist_sub))

            for rank, seg in enumerate(hist_recent):
                if not isinstance(seg, list) or len(seg) < 2:
                    continue
                s, e = int(seg[0]), int(seg[1])
                s = max(0, min(s, L)); e = max(0, min(e, L))
                if e <= s:
                    continue

                keep_ratio = float(ratios[rank] if rank < len(ratios) else ratios[-1])
                keep_ratio = 0.0 if keep_ratio < 0.0 else (1.0 if keep_ratio > 1.0 else keep_ratio)

                seg_ids = input_ids[b, s:e]
                seg_valid = (seg_ids != pad_token_id)
                if not torch.any(seg_valid):
                    continue

                rel_valid = torch.nonzero(seg_valid, as_tuple=False).squeeze(-1)
                abs_valid = (rel_valid + s).to(device)
                N = int(abs_valid.numel())
                if N <= 0:
                    continue

                # Determine keep budget
                keep_n = int((keep_ratio * N) // 1)
                if keep_n <= 0:
                    # drop all valid tokens in this history segment
                    keep_mask[b, abs_valid] = False
                    continue
                if keep_n >= N:
                    # keep all
                    continue

                # segment embeddings [N, H]
                emb = hidden_states[b, abs_valid, :]
                # pivots: top-L2
                norms = emb.norm(dim=-1)
                p = min(num_pivots, N)
                _, piv_rel = torch.topk(norms, k=p, largest=True)

                # cosine similarity to pivots
                emb_n = F.normalize(emb, p=2, dim=-1)
                piv_n = emb_n[piv_rel, :]
                sim = torch.matmul(emb_n, piv_n.t())  # [N, p]
                max_sim, _ = sim.max(dim=-1)  # [N]

                # keep pivots always; select remaining farthest tokens by smallest max_sim
                piv_mask = torch.zeros(N, dtype=torch.bool, device=device)
                piv_mask[piv_rel] = True

                remaining = keep_n - int(p)
                if remaining <= 0:
                    # only keep pivots (truncate to keep_n if keep_n < p)
                    if keep_n < p:
                        # choose subset of pivots with smallest max_sim among pivots (stable)
                        piv_scores = max_sim[piv_rel]
                        _, sub_rel = torch.topk(piv_scores, k=keep_n, largest=False)
                        keep_rel = piv_rel[sub_rel]
                    else:
                        keep_rel = piv_rel
                else:
                    cand_rel = torch.nonzero(~piv_mask, as_tuple=False).squeeze(-1)
                    if cand_rel.numel() == 0:
                        keep_rel = piv_rel
                    else:
                        cand_scores = max_sim[cand_rel]
                        take = min(remaining, int(cand_rel.numel()))
                        _, pick = torch.topk(cand_scores, k=take, largest=False)
                        keep_rel = torch.cat([piv_rel, cand_rel[pick]], dim=0)

                # apply to keep_mask: set whole segment False, then enable selected
                keep_mask[b, s:e] = False
                keep_abs = abs_valid[keep_rel]
                keep_mask[b, keep_abs] = True

        return keep_mask

    def _sparsevlm_build_keep_mask(self, *, hidden_states, hs_ln, attn_probs, OAS_index, input_ids, keep_ratio: float, tau: float, cluster_ratio: float):
        """
        SparseVLM 剪枝逻辑实现 (Stage 1, 2 & 3):

        Stage 1 (Raters 选择):
        - 使用 hidden_states 的点乘相似度
        - 动态阈值 m = mean(r)，选择 r >= m 的文本 token 作为 raters

        Stage 2 (视觉打分与初步划分):
        - 复用 attn_probs [B, heads, Q, K]
        - score[k] = sum_{q in raters} sum_{h} attn_probs[h, q, k]
        - 划分 V_keep40 (前 40%) 和 V_drop60 (后 60%)

        Stage 3 (回收、聚类与重构映射):
        - 从 V_drop60 中回收评分最高的 tau% (30%) 的 token
        - 使用 DPC 算法 (基于 hs_ln 的欧氏距离) 对回收池聚类
        - 聚类中心数 C = cluster_ratio * |V_drop60|
        - 缓存重构映射：簇内 token 索引 -> 聚类中心索引
        """
        B, L, H = hidden_states.shape
        device = hidden_states.device
        keep_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        self._sparsevlm_reconstructed_info = []

        if attn_probs is None or hs_ln is None:
            return keep_mask

        q_ranges = getattr(self, '_prompt_query_ranges', None)
        if q_ranges is None:
            return keep_mask

        _, n_heads, Q, K = attn_probs.shape

        for b in range(B):
            # 1) 获取文本范围 (p1+p2)
            item = q_ranges[b] if isinstance(q_ranges, list) and b < len(q_ranges) else q_ranges
            q_indices = []
            for key in ['p1', 'p2']:
                if isinstance(item, dict) and key in item and isinstance(item[key], (list, tuple)) and len(item[key]) == 2:
                    s_q, e_q = int(item[key][0]), int(item[key][1])
                    s_q, e_q = max(0, min(s_q, Q)), max(0, min(e_q, Q))
                    if e_q > s_q:
                        q_indices.extend(range(s_q, e_q))
            if not q_indices:
                self._sparsevlm_reconstructed_info.append(None)
                continue

            # 2) 获取历史视觉范围
            idx_list = OAS_index[b] if OAS_index is not None else []
            if not isinstance(idx_list, list) or len(idx_list) <= 1:
                self._sparsevlm_reconstructed_info.append(None)
                continue
            v_indices = []
            for seg in idx_list[:-1]:
                if isinstance(seg, list) and len(seg) >= 2:
                    s_v, e_v = int(seg[0]), int(seg[1])
                    s_v, e_v = max(0, min(s_v, K)), max(0, min(e_v, K))
                    if e_v > s_v:
                        v_indices.extend(range(s_v, e_v))
            if not v_indices:
                self._sparsevlm_reconstructed_info.append(None)
                continue

            # ===== Stage 1: 筛选 raters =====
            H_q = hidden_states[b, q_indices, :]
            H_v = hidden_states[b, v_indices, :]
            sim_vq = torch.matmul(H_v, H_q.transpose(-1, -2)) / math.sqrt(H)
            P = F.softmax(sim_vq, dim=-1)
            r = P.mean(dim=0)
            m = r.mean()
            raters_rel_idx = torch.nonzero(r >= m, as_tuple=False).squeeze(-1)
            if raters_rel_idx.numel() == 0:
                self._sparsevlm_reconstructed_info.append(None)
                continue
            raters_abs_q = torch.as_tensor([q_indices[i] for i in raters_rel_idx.tolist()], device=device, dtype=torch.long)

            # ===== Stage 2: 视觉打分 =====
            attn_sub = attn_probs[b, :, raters_abs_q, :]
            attn_sub = attn_sub[:, :, torch.as_tensor(v_indices, device=device, dtype=torch.long)]
            v_scores = attn_sub.sum(dim=(0, 1)) # [L_v]
            
            L_v = int(v_scores.numel())
            keep_k = max(1, int(keep_ratio * L_v))
            
            # 排序并划分
            vals, sorted_idx = torch.sort(v_scores, descending=True)
            v_indices_tensor = torch.as_tensor(v_indices, device=device, dtype=torch.long)
            
            keep40_rel = sorted_idx[:keep_k]
            drop60_rel = sorted_idx[keep_k:]
            
            keep_mask[b, v_indices] = False
            keep_mask[b, v_indices_tensor[keep40_rel]] = True

            # ===== Stage 3: 回收、聚类与重构映射 =====
            if drop60_rel.numel() == 0:
                self._sparsevlm_reconstructed_info.append(None)
                continue

            # 3.1 回收池：取被删部分中评分最高的 tau%
            num_recycle = max(1, int(tau * drop60_rel.numel()))
            recycle_rel = drop60_rel[:num_recycle]
            recycle_abs = v_indices_tensor[recycle_rel]
            
            # 3.2 聚类参数
            cluster_num = max(1, int(cluster_ratio * drop60_rel.numel()))
            cluster_num = min(cluster_num, recycle_abs.numel())
            k_knn = cluster_num
            
            # 3.3 DPC 聚类逻辑 (使用 hs_ln)
            H_recycle_ln = hs_ln[b, recycle_abs, :] # [N_rec, H]
            dist_mat = torch.cdist(H_recycle_ln, H_recycle_ln, p=2).pow(2) # [N_rec, N_rec]
            
            # 局部密度 rho
            if k_knn < dist_mat.size(0):
                knn_dist, _ = torch.topk(dist_mat, k=k_knn+1, largest=False, dim=-1)
                rho = torch.exp(-knn_dist[:, 1:].mean(dim=-1))
            else:
                rho = torch.exp(-dist_mat.mean(dim=-1))
            
            # 距离指示量 delta
            N_rec = recycle_abs.numel()
            delta = torch.zeros_like(rho)
            rho_max_idx = torch.argmax(rho)
            for i in range(N_rec):
                higher_rho_idx = torch.nonzero(rho > rho[i], as_tuple=False).squeeze(-1)
                if higher_rho_idx.numel() > 0:
                    delta[i] = dist_mat[i, higher_rho_idx].min()
                else:
                    delta[i] = dist_mat[i].max()
            
            # 选中心：rho * delta
            center_scores = rho * delta
            _, center_rec_idx = torch.topk(center_scores, k=cluster_num, largest=True)
            center_abs = recycle_abs[center_rec_idx]
            
            # 分配其余回收 token (余弦相似度)
            H_centers_ln = H_recycle_ln[center_rec_idx]
            H_rec_norm = F.normalize(H_recycle_ln, p=2, dim=-1)
            H_centers_norm = F.normalize(H_centers_ln, p=2, dim=-1)
            cos_sim = torch.matmul(H_rec_norm, H_centers_norm.transpose(-1, -2)) # [N_rec, cluster_num]
            assign_idx = torch.argmax(cos_sim, dim=-1) # 每个回收 token 分配到的中心索引 (0 ~ cluster_num-1)
            
            # 记录重构信息：簇索引 -> [成员 abs 索引列表], 中心 abs 索引
            reconstruct_map = []
            for c in range(cluster_num):
                member_mask = (assign_idx == c)
                member_abs = recycle_abs[member_mask]
                reconstruct_map.append({
                    'center_abs': int(center_abs[c].item()),
                    'members_abs': member_abs.tolist()
                })
            
            # 更新 keep_mask：中心 token 必须保留
            keep_mask[b, center_abs] = True
            self._sparsevlm_reconstructed_info.append(reconstruct_map)

        return keep_mask

    def Drop_hiddenstates_sparsevlm(self, hidden, keep_mask, input_ids, position_embeddings, reconstructed_info):
        """
        SparseVLM 专用裁剪函数：
        1. 先在原位用簇内 hidden_states (未LN) 求和替换中心 token
        2. 然后执行正常的 mask 裁剪
        """
        hidden_states = hidden.clone()
        B, L, H = hidden_states.shape
        
        # Step 1: 原位替换中心
        if reconstructed_info is not None:
            for b in range(min(B, len(reconstructed_info))):
                info = reconstructed_info[b]
                if info is None: continue
                for cluster in info:
                    c_idx = cluster['center_abs']
                    m_idxs = cluster['members_abs']
                    if m_idxs:
                        # 逐元素求和 (公式 10)
                        sum_vec = hidden[b, m_idxs, :].sum(dim=0)
                        hidden_states[b, c_idx, :] = sum_vec
        
        # Step 2: 调用现有的裁剪逻辑
        return self.Drop_hiddenstates_fastv(hidden_states, keep_mask, input_ids, position_embeddings)

    def _DivPrune_selection(
            self,
            indices_abs: torch.Tensor,
            hidden_states_b: torch.Tensor,
            target_n: int,
            device: torch.device,
    ) -> torch.Tensor:
        """
        在给定的绝对索引集合内进行贪心选择（Max-Min 距离最大化）。
        目标：每轮选择与当前已选集合“最小距离”最大的 token。
        距离定义：d = 1 - cos，相当于最小化 max(cos) 的候选。

        初始化策略：
        - 在段内计算每个 token 到其最近邻（排除自身）的距离；
        - 选择“最近邻距离最大”的 token 作为首个保留点（seed）。
        """
        N = indices_abs.numel()
        if target_n <= 0:
            return indices_abs.new_zeros((0,))
        if target_n >= N:
            return indices_abs

        # 提取 embedding 并计算余弦相似度矩阵 [N, N]
        emb = hidden_states_b[indices_abs]
        emb = F.normalize(emb, p=2, dim=-1)
        sim_mat = torch.mm(emb, emb.t()).clamp(-1.0, 1.0)

        # 初始化掩码和计数
        selected_mask = torch.zeros(N, dtype=torch.bool, device=device)

        # 初始化：选择“最近邻距离最大”的 token 作为 seed
        # 距离定义 d = 1 - cos，排除自身（对角线置为 +inf）
        dist_mat = 1.0 - sim_mat
        dist_mat.fill_diagonal_(float('inf'))
        nn_dist = dist_mat.min(dim=1).values
        seed_rel = torch.argmax(nn_dist).view(1)

        selected_mask[seed_rel] = True
        num_selected = 1

        # 记录每个候选到已选集合的“最大余弦相似度” max_sim_to_selected
        # 对应最小距离：min_dist = 1 - max_sim_to_selected
        max_sim_to_selected = sim_mat[:, seed_rel].max(dim=1).values

        # 贪心循环（固定逐个选择）
        step_k = 1

        while num_selected < target_n:
            k_this = min(step_k, target_n - num_selected)

            # 选择“最小距离最大”等价于“最大相似度最小”的候选
            # 已选位置填充 +inf，避免重复选择
            candidate_scores = max_sim_to_selected.masked_fill(selected_mask, float('inf'))

            if k_this == 1:
                best_rel = torch.argmin(candidate_scores).view(1)
            else:
                _, best_rel = torch.topk(candidate_scores, k=k_this, largest=False)

            # 更新：并行加入后，维护每个候选到新已选集合的最大相似度
            selected_mask[best_rel] = True
            max_sim_to_selected = torch.maximum(max_sim_to_selected, sim_mat[:, best_rel].max(dim=1).values)
            num_selected += int(k_this)

        # 返回最终选中的绝对索引
        return indices_abs[selected_mask]

    def _DivPrune_build_keep_mask(
            self,
            hidden_states: torch.Tensor,  # [B, L, H]
            OAS_index,
            DivPrune_info,
            input_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, L, H = hidden_states.shape
        device = hidden_states.device
        pad_token_id = 151643
        keep_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        # 不再依赖 Sobel 先验：直接在每张历史截图段内部进行整体贪心选择

        for b in range(B):
            idx_list = OAS_index[b] if OAS_index is not None else []
            hist_segments = []
            for seg in idx_list[:-1]:
                if isinstance(seg, list) and len(seg) >= 2:
                    s, e = int(seg[0]), int(seg[1])
                    if e > s and s >= 0 and e <= L:
                        hist_segments.append((s, e))

            if not hist_segments:
                continue

            T = len(hist_segments)

            for img_id, (s, e) in enumerate(hist_segments):
                # 有效 token（排除 pad）
                seg_valid_mask = (input_ids[b, s:e] != pad_token_id)
                rel_valid = torch.nonzero(seg_valid_mask, as_tuple=False).squeeze(-1)
                abs_valid = (rel_valid + s).to(device)
                n_valid = int(abs_valid.numel())
                if n_valid == 0:
                    continue

                # 确定该图片的保留比例 budget_ratio (K)
                if self.DivPrune_ratios and len(self.DivPrune_ratios) > 0:
                    dist = (T - 1) - img_id
                    budget_ratio = self.DivPrune_ratios[dist] if dist < len(self.DivPrune_ratios) else self.DivPrune_ratios[-1]
                else:
                    budget_ratio = 1.0

                budget_ratio = 0.0 if budget_ratio < 0.0 else (1.0 if budget_ratio > 1.0 else budget_ratio)
                keep_n = int(budget_ratio * n_valid)
                if keep_n <= 0:
                    keep_n = 1

                # 段内整体贪心筛选
                keep_abs = self._DivPrune_selection(abs_valid, hidden_states[b], keep_n, device)

                # 先把整张图段设为 False，再把选中的设为 True（pad 位置不参与 abs_valid，不受影响）
                keep_mask[b, s:e] = False
                if keep_abs.numel() > 0:
                    keep_mask[b, keep_abs] = True

        return keep_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            OAS_index=None,
            input_ids_aux=None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # drop layers from 3
        for i, decoder_layer in enumerate(self.layers):
            if i == 0 and hidden_states.shape[1] > 1:
                self._n_before_prefill = int(hidden_states.shape[1])
                # Initialize n_after with n_before in case no pruning happens
                self._n_after_hook = int(hidden_states.shape[1])

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Step 1: 当启用剪枝时，在第 drop_k-1 层收集信息
            # 统一缓存 OAS_index 供后续可视化使用。
            # 增加判定：仅当找到图像段时才更新，避免被 Decoding 阶段的空输入覆盖。
            if (
                    self.enable_DivPrune_pruning or self.enable_dart_pruning or self.enable_fastv_pruning or self.enable_pdrop_pruning or self.enable_uigraph_pruning or self.sobel_enable or self.enable_random_pruning or self.enable_sparsevlm_pruning) and (
                    i == (self.drop_k - 1)):
                if OAS_index is not None and any(len(sample_idx) > 0 for sample_idx in OAS_index):
                    self._last_oas_index = OAS_index

            # Sparse Greedy: 直接使用hidden_states，不需要注意力
            if self.enable_DivPrune_pruning and (i == (self.drop_k - 1)) and hidden_states.shape[1] != 1:
                try:
                    keep_mask_sg = self._DivPrune_build_keep_mask(
                        hidden_states=hidden_states,
                        OAS_index=OAS_index,
                        DivPrune_info=self._DivPrune_info,
                        input_ids=input_ids_aux,
                    )
                    # 缓存本次推理的mask与段索引，供外部脚本统计各历史截图段的保留token比例
                    self._DivPrune_keep_mask = keep_mask_sg.detach()
                except Exception as e:
                    logger.warning_once(f"Sparse Greedy: failed to build keep_mask at layer {i}: {e}")
                    self._DivPrune_keep_mask = None
            # DART/FastV/UIgraph/Sobel: 需要收集注意力概率或构造基于 hidden_states 的 keep_mask
            elif (
                    self.enable_dart_pruning or self.enable_fastv_pruning or self.enable_pdrop_pruning or self.enable_uigraph_pruning or self.sobel_enable or self.enable_random_pruning or self.enable_sparsevlm_pruning) and (
                    i == (self.drop_k - 1)) and hidden_states.shape[1] != 1:
                try:
                    attn_probs = self._fastv_compute_attn_probs(
                        decoder_layer=decoder_layer,
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    self._fastv_last_attn_weights = attn_probs.detach()
                    # 模式互斥：优先 Sobel > UIgraph > DART > FastV > Random > SparseVLM
                    # 注意：可视化依赖 keep_mask，在 decoding 阶段（seq_len==1）不应覆盖 prefill 阶段的缓存。
                    if hidden_states.shape[1] != 1:
                        self._sobel_keep_mask = None
                        self._uigraph_keep_mask = None
                        self._fastv_keep_mask = None
                        self._pdrop_keep_mask = None
                        self._random_keep_mask = None
                        self._dart_keep_mask = None
                        self._sparsevlm_keep_mask = None
                    if self.sobel_enable and (self._sobel_info is not None):
                        try:
                            keep_mask_sobel = self._sobel_build_keep_mask(OAS_index, self._sobel_info, input_ids_aux,
                                                                          pad_token_id=151643)
                            self._sobel_keep_mask = keep_mask_sobel.detach()
                        except Exception as e:
                            logger.warning_once(f"Sobel(edge-only): failed to build keep_mask at layer {i}: {e}")
                            self._sobel_keep_mask = None
                    elif self.enable_uigraph_pruning and (self._uigraph_info is not None):
                        try:
                            keep_mask_ui = self._uigraph_build_keep_mask(attn_probs, OAS_index, self._uigraph_info,
                                                                         input_ids_aux)
                            self._uigraph_keep_mask = keep_mask_ui.detach()
                        except Exception as e:
                            logger.warning_once(f"UIgraph: failed to build keep_mask at layer {i}: {e}")
                            self._uigraph_keep_mask = None
                    elif self.enable_dart_pruning:
                        try:
                            keep_mask_dart = self._dart_build_keep_mask(
                                hidden_states=hidden_states,
                                OAS_index=OAS_index,
                                input_ids=input_ids_aux,
                                pad_token_id=151643,
                            )
                            self._dart_keep_mask = keep_mask_dart.detach()
                            if bool(getattr(self, 'dart_debug', False)):
                                keep_cnt = int(self._dart_keep_mask.sum().item())
                                total_cnt = int(self._dart_keep_mask.numel())
                                logger.warning_once(
                                    f"[DART-DEBUG][layer={i}] built dart keep-mask: keep={keep_cnt}/{total_cnt}"
                                )
                        except Exception as e:
                            logger.warning_once(f"DART: failed to build keep_mask at layer {i}: {e}")
                            self._dart_keep_mask = None
                    elif self.enable_fastv_pruning:
                        try:
                            ratios = getattr(self, 'fastv_segment_keep_ratios', None)
                            if not (isinstance(ratios, (list, tuple)) and len(ratios) > 0):
                                # backward compatibility: old prune-ratio semantics
                                old_ratios = getattr(self, 'fastv_segment_ratios', None)
                                if isinstance(old_ratios, (list, tuple)) and len(old_ratios) > 0:
                                    ratios = [1.0 - float(x) for x in old_ratios]
                            if not (isinstance(ratios, (list, tuple)) and len(ratios) > 0):
                                raise ValueError("FastV requires non-empty fastv_segment_keep_ratios")
                            keep_mask = self._fastv_build_keep_mask_per_image(attn_probs, OAS_index, input_ids_aux,
                                                                              ratios, pad_token_id=151643)
                            self._fastv_keep_mask = keep_mask.detach()
                        except Exception as e:
                            logger.warning_once(f"FastV: failed to build keep_mask at layer {i}: {e}")
                            self._fastv_keep_mask = None
                    elif self.enable_pdrop_pruning:
                        try:
                            ratios = getattr(self, 'pdrop_keep_ratios', None)
                            if not (isinstance(ratios, (list, tuple)) and len(ratios) > 0):
                                # backward compatibility: old single-ratio semantics
                                old_keep_r = float(getattr(self, 'pdrop_keep_ratio_m', 0.5))
                                old_keep_r = 0.0 if old_keep_r < 0.0 else (1.0 if old_keep_r > 1.0 else old_keep_r)
                                ratios = [old_keep_r]
                            qidx = getattr(self, '_pdrop_query_index', None)
                            keep_mask_pdrop = self._pdrop_build_keep_mask(
                                attn_probs=attn_probs,
                                OAS_index=OAS_index,
                                input_ids=input_ids_aux,
                                ratios=ratios,
                                query_index=qidx,
                                pad_token_id=151643,
                            )
                            self._pdrop_keep_mask = keep_mask_pdrop.detach()
                        except Exception as e:
                            logger.warning_once(f"PDrop: failed to build keep_mask at layer {i}: {e}")
                            self._pdrop_keep_mask = None
                    elif self.enable_random_pruning:
                        try:
                            keep_mask_random = self._random_build_keep_mask(OAS_index, input_ids_aux,
                                                                            pad_token_id=151643)
                            self._random_keep_mask = keep_mask_random.detach()
                        except Exception as e:
                            logger.warning_once(f"Random: failed to build keep_mask at layer {i}: {e}")
                            self._random_keep_mask = None
                    elif self.enable_sparsevlm_pruning:
                        try:
                            hs_ln = decoder_layer.input_layernorm(hidden_states)
                            keep_mask_sparsevlm = self._sparsevlm_build_keep_mask(
                                hidden_states=hidden_states,
                                hs_ln=hs_ln,
                                attn_probs=attn_probs,
                                OAS_index=OAS_index,
                                input_ids=input_ids_aux,
                                keep_ratio=float(self.sparsevlm_keep_ratio),
                                tau=float(self.sparsevlm_tau),
                                cluster_ratio=float(self.sparsevlm_cluster_ratio)
                            )
                            self._sparsevlm_keep_mask = keep_mask_sparsevlm.detach()
                        except Exception as e:
                            logger.warning_once(f"SparseVLM: failed to build keep_mask at layer {i}: {e}")
                            self._sparsevlm_keep_mask = None
                except Exception as e:
                    logger.warning_once(f"Sobel/UIgraph/FastV/SparseVLM: failed to collect attn at layer {i}: {e}")
                    self._fastv_last_attn_weights = None

            # If inference
            # 说明：
            # - 该分支仅在推理阶段（not self.training）生效，用于在第 drop_k 层触发一次序列裁剪/压缩。
            # - 触发条件：当前层索引 i == self.drop_k 且当前序列长度 hidden_states.shape[1] != 1。
            #   这意味着仅在“prefill”阶段（序列长度>1）执行裁剪，避免在逐 token 解码（长度=1）时破坏生成流程。
            # - 核心操作：调用 Drop_hiddenstates(...) 对以下三者做一致性裁剪：
            #     * hidden_states：按保留的 token 子集重排/截断，减少后续层的计算量；
            #     * position_embeddings：同步到裁剪后的 token 序列，以保持位置对齐；
            #     * padded_attention_mask：同步更新 padding/可见性，确保注意力矩阵与新序列一致。
            # - 掩码更新：裁剪后需重建因果掩码（causal_mask），分两种实现路径：
            #     * 非 flash_attention_2：使用 _update_causal_mask_compression(...)，针对压缩后变长/不规则序列的专用掩码逻辑；
            #     * 使用 flash_attention_2：调用 _update_causal_mask(...)，遵循 FA2 的接口，依赖 cache_position 与 past_key_values。
            # - 目的：在指定层（drop_k）减少有效 token 数，使后续层以更少的序列长度进行计算，从而加速推理，同时尽量保持性能。
            if not self.training:
                if i == self.drop_k and hidden_states.shape[1] != 1:
                    # 模式优先：Sparse Greedy > Sobel > UIgraph > DART > FastV > Random > SparseVLM > 原始
                    if bool(getattr(self, 'dart_debug', False)):
                        _mode = "FallbackDropAll"
                        if self.enable_DivPrune_pruning and (self._DivPrune_keep_mask is not None):
                            _mode = "DivPrune"
                        elif self.sobel_enable and (self._sobel_keep_mask is not None):
                            _mode = "Sobel"
                        elif self.enable_uigraph_pruning and (self._uigraph_keep_mask is not None):
                            _mode = "UIgraph"
                        elif self.enable_dart_pruning and (self._dart_keep_mask is not None):
                            _mode = "DART"
                        elif self.enable_fastv_pruning and (self._fastv_keep_mask is not None):
                            _mode = "FastV"
                        elif self.enable_pdrop_pruning and (self._pdrop_keep_mask is not None):
                            _mode = "PDrop"
                        elif self.enable_random_pruning and (self._random_keep_mask is not None):
                            _mode = "Random"
                        elif self.enable_sparsevlm_pruning and (self._sparsevlm_keep_mask is not None):
                            _mode = "SparseVLM"
                        logger.warning_once(
                            f"[DART-DEBUG][drop@layer={i}] selected_mode={_mode} "
                            f"has_masks(dart={self._dart_keep_mask is not None}, fastv={self._fastv_keep_mask is not None}, "
                            f"pdrop={self._pdrop_keep_mask is not None}, random={self._random_keep_mask is not None}, "
                            f"sobel={self._sobel_keep_mask is not None}, uigraph={self._uigraph_keep_mask is not None}, "
                            f"divprune={self._DivPrune_keep_mask is not None}, sparsevlm={self._sparsevlm_keep_mask is not None})"
                        )
                    if self.enable_DivPrune_pruning and (self._DivPrune_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_fastv(
                            hidden_states, self._DivPrune_keep_mask, input_ids_aux, position_embeddings
                        )
                    elif self.sobel_enable and (self._sobel_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_fastv(
                            hidden_states, self._sobel_keep_mask, input_ids_aux, position_embeddings
                        )
                    elif self.enable_uigraph_pruning and (self._uigraph_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_UIgraph(
                            hidden_states, self._uigraph_keep_mask, input_ids_aux, position_embeddings
                        )
                    elif self.enable_dart_pruning and (self._dart_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_fastv(
                            hidden_states, self._dart_keep_mask, input_ids_aux, position_embeddings
                        )
                    elif self.enable_fastv_pruning and (self._fastv_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_fastv(
                            hidden_states, self._fastv_keep_mask, input_ids_aux, position_embeddings
                        )
                    elif self.enable_pdrop_pruning and (self._pdrop_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_fastv(
                            hidden_states, self._pdrop_keep_mask, input_ids_aux, position_embeddings
                        )
                    elif self.enable_random_pruning and (self._random_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_fastv(
                            hidden_states, self._random_keep_mask, input_ids_aux, position_embeddings
                        )
                    elif self.enable_sparsevlm_pruning and (self._sparsevlm_keep_mask is not None):
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates_sparsevlm(
                            hidden_states, self._sparsevlm_keep_mask, input_ids_aux, position_embeddings, self._sparsevlm_reconstructed_info
                        )
                    else:
                        hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates(
                            hidden_states, OAS_index, input_ids_aux, position_embeddings
                        )
                    if self._attn_implementation != "flash_attention_2":
                        causal_mask = self._update_causal_mask_compression(
                            padded_attention_mask, hidden_states, output_attentions
                        )
                    else:
                        causal_mask = self._update_causal_mask(
                            padded_attention_mask, hidden_states, cache_position, past_key_values, output_attentions
                        )

                    # Update FLOPs hook with the sequence length after pruning (prefill only)
                    self._n_after_hook = int(hidden_states.shape[1])
            # If training
            else:
                if i == self.drop_k:
                    hidden_states, position_embeddings, padded_attention_mask = self.Drop_hiddenstates(hidden_states,
                                                                                                       OAS_index,
                                                                                                       input_ids_aux,
                                                                                                       position_embeddings)
                    causal_mask = self._update_causal_mask(
                        padded_attention_mask, hidden_states, cache_position, past_key_values, output_attentions
                    )

                # causal_mask = self.Drop_causal_mask(hidden_states, OAS_index, input_ids_aux)
                # position_embeddings = self.Drop_position_embeddings(hidden_states, OAS_index, input_ids_aux)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _update_causal_mask_compression(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            # cache_position: torch.Tensor,
            # target_len: Cache,
            output_attentions: bool,
    ):
        cache_position = torch.arange(attention_mask.shape[-1]).to(attention_mask.device)
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        # past_seen_tokens = target_len if target_len is not None else 0
        # using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        # if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
        #     if AttentionMaskConverter._ignore_causal_mask_sdpa(
        #         attention_mask,
        #         inputs_embeds=input_tensor,
        #         past_key_values_length=past_seen_tokens,
        #         is_training=self.training,
        #     ):
        #         return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # if using_static_cache:
        #     target_length = past_key_values.get_max_length()
        # else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else 0 + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


QWEN2_VL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        pixel_values (`torch.FloatTensor` of shape `(seq_length, num_channels * image_size * image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing images.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
"""


class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        # self.summary = nn.Parameter(torch.randn(64, 1536))
        self.drop_k = 3
        self.model = Qwen2VLModel(config)
        self.model.drop_k = self.drop_k
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        # torch.nn.init.xavier_uniform_(self.summary)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
            self,
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    # def Replace_summary(self, input_embs, input_ids):
    #     input_embs = input_embs.clone()
    #     all_index = find_indices_in_batch(input_ids)
    #     for i, index in enumerate(all_index):
    #         if len(index) != 0:
    #             for his in index:
    #                 # input_embs[i] = torch.concat((input_embs[i, :his[0]],self.summary,input_embs[i, his[0] + 64:]),dim=0)
    #                 input_embs[i, his[0]: his[0] + 64] = self.summary
    #     return input_embs

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        ## View input_ids, targets
        # from transformers import AutoProcessor

        # processor = AutoProcessor.from_pretrained("/nas_sh/wentao/Qwen2-VL-7B-Instruct/")
        # tokenizer = processor.tokenizer
        # print("--------------------inputids-------------------")
        # demo_input_ids = input_ids[0]
        # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(demo_input_ids)))

        # print("--------------------targets-------------------")
        # demo_labels = labels[0]
        # mask = demo_labels != -100
        # demo_labels = demo_labels[mask]
        # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(demo_labels)))

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # replace the summary id to summary params
            # inputs_embeds = self.Replace_summary(inputs_embeds, input_ids)
            OAS_index = Find_OAS(input_ids, self.config.image_token_id)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

        # print("OAS_index", OAS_index)
        # Prefill should include both: (1) no cache object, and (2) empty cache object (seq_len == 0)
        is_prefill_pass = past_key_values is None
        if not is_prefill_pass and hasattr(past_key_values, "get_seq_length"):
            try:
                is_prefill_pass = past_key_values.get_seq_length() == 0
            except Exception:
                pass
        prefill_t0 = time.perf_counter() if is_prefill_pass else None
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            OAS_index=OAS_index,
            input_ids_aux=input_ids
        )
        if is_prefill_pass and prefill_t0 is not None:
            self.model._last_prefill_time_ms = float((time.perf_counter() - prefill_t0) * 1000.0)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # process labels
            pad_after = logits.shape[1]
            padded_label = []

            for i, label in enumerate(labels):
                cur_label = []
                cur_index = OAS_index[i]
                for j, index in enumerate(cur_index):
                    if index != []:
                        if j == 0:
                            cur_label.append(label[0:index[0]])
                        else:
                            cur_label.append(label[cur_index[j - 1][1]:cur_index[j][0]])

                        if j == (len(cur_index) - 2):
                            cur_label.append(label[cur_index[j][1]:])

                    else:
                        # no history case
                        if j == 0:
                            cur_label.append(label)

                cur_label = torch.concat(cur_label, dim=0)

                if len(cur_label) < pad_after:
                    padding_length = pad_after - len(cur_label)

                    # 创建值全为 -100 的 Tensor
                    padding_tensor = torch.full((padding_length,), -100).to(cur_label.dtype).to(cur_label.device)
                    cur_label = torch.concat((cur_label, padding_tensor), dim=0)

                elif len(cur_label) > pad_after:
                    cur_label = cur_label[0:pad_after]

                padded_label.append(cur_label)
            padded_label = torch.stack(padded_label, dim=0)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = padded_label[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs
