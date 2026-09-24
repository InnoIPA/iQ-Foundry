# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#  NOT A CONTRIBUTION.

# ==============================================================================
# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
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
"""
Integration with GGML / The file is copied and adapted from https://github.com/99991/pygguf
with extra methods beings exposed
"""

"""
GGUF to QNN offset conversion

Notation
--------
- bw           : bit-width of the integer code (e.g., 2, 3, 4, 5, 8)
- midpoint     : midpoint constant = 2^(bw - 1)
- quant_val    : quantized weight
- float_val    : real-valued weight
- scale        : per-block scale
- gguf_offset  : GGUF offset term
- qnn_offset   : QNN offset term

QNN (asymmetric) dequantization
-------------------------------
    float_val = (quant_val + qnn_offset) * scale                                  (1)

For weight quantization in QNN, the range is signed [-midpoint, midpoint - 1].
But, the range of quant_val in asymmetric GGUF dataypes is typically in the 
unsigned integer range [0, 2^bw - 1].

To align this with QNN, we subtract the midpoint of unsgined range from quant_val in GGUF:
    signed_val = quant_val - midpoint  (signed_val now matches the effective quant_val in QNN domain in (1))
    
    quant_val = signed_val + midpoint                                             (2)

GGUF dequantization variants
----------------------------
1) K-formats (asymmetric, subtractive offset): Q4_K / Q2_K / Q5_K
       float_val = (quant_val * scale) - gguf_offset                              (3)
       Substitute (2) into (3):
       float_val = ((signed_val + midpoint) * scale) - gguf_offset
       float_val = (signed_val * scale) + (midpoint * scale) - gguf_offset        (4)

   Comparing (1) and (4):
    (quant_val + qnn_offset) * scale = (signed_val * scale) + (midpoint * scale) - gguf_offset
    qnn_offset * scale = (midpoint * scale) - gguf_offset   
    qnn_offset = midpoint - (gguf_offset / scale)                                   
   

2) Q4_1 (asymmetric, additive offset)
       float_val = (quant_val * scale) + gguf_offset                              (5)
       Substitute (2) into (5):
       float_val = ((signed_val + midpoint) * scale) + gguf_offset
       float_val = (signed_val * scale) + (midpoint * scale) + gguf_offset        (6)
    
   Comparing (1) and (6):
    (quant_val + qnn_offset) * scale = (signed_val * scale) + (midpoint * scale) + gguf_offset
    qnn_offset * scale = (midpoint * scale) + gguf_offset   
    qnn_offset = midpoint + (gguf_offset / scale)

3) Symmetric formats (no offset): Q4_0 / Q6_K / Q8_0 / Q3_K
       float_val = quant_val * scale
       gguf_offset = 0
"""

from array import array
from typing import Optional

import numpy as np
from tokenizers import Tokenizer, Regex, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram
from typing import Tuple, Dict, List

from .. import AddedToken
from ..convert_slow_tokenizer import Converter, SpmConverter, SentencePieceExtractor, generate_merges, LlamaConverter, Qwen2Converter
from ..utils import logging
from ..utils.logging import tqdm


logger = logging.get_logger(__name__)

ENABLE_ASYMMETRIC_MIXED_PRECISION = True

def enable_asym_mp(is_asym_mp: bool):
    global ENABLE_ASYMMETRIC_MIXED_PRECISION
    ENABLE_ASYMMETRIC_MIXED_PRECISION = is_asym_mp

# Listed here: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
}

# The Blocksizes are reported in bytes
# Check out: https://github.com/ggerganov/llama.cpp/blob/8a56075b07a8b571bf95a912ffdce4c928c2b414/gguf-py/gguf/constants.py#L801
GGML_BLOCK_SIZES = {
    "Q8_0": 2 + 32,  # Q8_0 uses a blocksize of 32 (int8 tensors) + 2 bytes allocated for the scales
    "Q4_K": 144, # 256/2 (weights) + 2 (super-block scale) + 2 (super-block min) + 8*6/8 (block scales) + 8*6/8 (block min)
    # Q4_0 uses a blocksize of 32 but the 4-bit tensors are packed into 8-bit tensors + 2 bytes for the scales
    "Q4_0": 2 + 16,
    "Q4_1": 2 + 2 + 16,
    "Q6_K": 210, # 256*6/8 (weights) + 16 (block scale) + 2 (super-block scale)
    # See: https://github.com/99991/pygguf/commit/a417edbfc029a1bc270f984a694f9128c5afa8b9
    "Q2_K": 256 // 16 + 256 // 4 + 2 + 2,
    "Q3_K": 256 // 8 + 256 // 4 + 12 + 2,
    "Q5_0": 2 + 4 + 16,
    "Q5_1": 2 + 2 + 4 + 16,
    "Q5_K": 2 + 2 + 12 + 256 // 8 + 256 // 2,
}

# Listed here: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
DATA_TYPES = {
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
}

GGUF_TENSOR_MAPPING_COMMON = {
    "token_embd": "model.embed_tokens",
    "blk": "model.layers",
    "ffn_down": "mlp.down_proj",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_output": "self_attn.o_proj",
    "output.weight": "lm_head.weight",
    "output_norm": "model.norm",
}

GGUF_TENSOR_MAPPING = {
    "llama": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.up_proj",
        "ffn_gate": "mlp.gate_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
    },
    "mistral": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.up_proj",
        "ffn_gate": "mlp.gate_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
    },
    "phi3": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.gate_up_proj",
        "ffn_gate": "mlp.gate_up_proj",
        "attn_qkv": "self_attn.qkv_proj",
    },
    "qwen2": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.up_proj",
        "ffn_gate": "mlp.gate_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
    },
    "qwen3": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.up_proj",
        "ffn_gate": "mlp.gate_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k_norm": "self_attn.k_norm",
        "attn_q_norm": "self_attn.q_norm",
    },
    "gemma": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.up_proj",
        "ffn_gate": "mlp.gate_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
    },
    "gemma2": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.up_proj",
        "ffn_gate": "mlp.gate_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffw_norm":"post_feedforward_layernorm",
        "post_attention_norm":"post_attention_layernorm",
    },
    "gemma3": {
        **GGUF_TENSOR_MAPPING_COMMON,
        "ffn_up": "mlp.up_proj",
        "ffn_gate": "mlp.gate_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffw_norm":"post_feedforward_layernorm",
        "post_attention_norm":"post_attention_layernorm",
        "attn_k_norm": "self_attn.k_norm",
        "attn_q_norm": "self_attn.q_norm",
    }
}

GGUF_CONFIG_MAPPING_COMMON = {
    "context_length": "max_position_embeddings",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "embedding_length": "hidden_size",
    "rope.dimension_count": None,
    "rope.freq_base": "rope_theta",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "vocab_size": "vocab_size",
}

GGUF_CONFIG_MAPPING = {
    "general": {
        "architecture": "model_type",
        "name": "_model_name_or_path",
    },

    "llama": {
        **GGUF_CONFIG_MAPPING_COMMON,
    },
    "mistral": {
        **GGUF_CONFIG_MAPPING_COMMON,
        "rope.dimension_count": "head_dim",
    },
    "phi3": {
        **GGUF_CONFIG_MAPPING_COMMON,
        "rope.scaling.original_context_length": "original_max_position_embeddings",
    },
    "qwen2": {
        **GGUF_CONFIG_MAPPING_COMMON,
    },
    "qwen3": {
        **GGUF_CONFIG_MAPPING_COMMON,
        "attention.key_length": "head_dim",
    },
    "gemma": {
        **GGUF_CONFIG_MAPPING_COMMON,
    },
    "gemma2": {
        **GGUF_CONFIG_MAPPING_COMMON,
        "attention.key_length": "head_dim",
        "attention.sliding_window": "sliding_window",
    },
    "gemma3": {
        **GGUF_CONFIG_MAPPING_COMMON,
        "attention.key_length": "head_dim",
        "attention.sliding_window": "sliding_window",
    },
    "tokenizer": {
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
}

GGUF_TOKENIZER_MAPPING = {
    "tokenizer": {
        "ggml.model": "tokenizer_type",
        "ggml.pre": "pre_tokenizer_type",
        "ggml.tokens": "tokens",
        "ggml.scores": "scores",
        "ggml.token_type": "token_type",
        "ggml.merges": "merges",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
        "ggml.add_space_prefix": "add_prefix_space",
    },
    "tokenizer_config": {
        "chat_template": "chat_template",
        "ggml.model": "model_type",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
}

MISTRAL_VARIANTS = ["mistral", "ministral", "mathstral"]

def _gguf_parse_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        if data_type[0] != 9:
            raise ValueError("Received multiple types, therefore expected the first type to indicate an array.")
        data_type, array_data_type = data_type

    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type in [9]:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value


def generate_encodings(block_size, bw, dtype, enc_type, is_sym, offset, scale):
    encodings = dict()
    encodings['block_size'] = block_size
    encodings['bw'] = bw
    encodings['dtype'] = dtype
    encodings['enc_type'] = enc_type
    encodings['is_sym'] = is_sym
    encodings['offset'] = offset.flatten().tolist()
    encodings['scale'] = scale.flatten().tolist()
    return encodings


def update_negative_scales_symmetric(values, scales, bitwidth):
    scaled_max_values = np.max(np.abs(values), axis=-1, keepdims=True) / (2**(bitwidth-1)-1)
    scales[scales < 0] = scaled_max_values[scales < 0]
    return scales


def dequantize_q4_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)

    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    factors = scale_factors * np.concatenate(
        [qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1
    )
    offsets = scale_offsets * np.concatenate(
        [qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1
    )

    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xF, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)
    # Dequantize final weights using scales and offsets
    values = factors * qs2 - offsets
    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        qnn_bitwidth = 4
        # Replace zeros in scales with 1
        updated_factors = np.where(factors == 0, 1.0, factors)
        qnn_offsets = np.ones(offsets.shape, dtype=np.float32) * (2**(qnn_bitwidth-1)) + (-offsets / updated_factors)
        # As per ggml-quant.c -> Q4_K has only positive scales, so no requirement to update negative scales
        encodings = generate_encodings(32, qnn_bitwidth, 'INT', 'PER_BLOCK', False, qnn_offsets, updated_factors)
    return values, encodings

def dequantize_q4_1(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1106
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L18
    block_size = GGML_BLOCK_SIZES["Q4_1"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # The scales are stored on the first 2 bytes,
    # The offsets are stored on the next 2 bytes,
    # and the rest corresponds to the quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    offsets = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    # the rest of the bytes corresponds to the quants - we discard the first four bytes
    quants = data_u8[:, 4:]

    ql = (quants[:, :] & 0xF).astype(np.uint8)
    qr = (quants[:, :] >> 4).astype(np.uint8)

    # Use hstack
    quants = np.hstack([ql, qr])
    values = (scales * quants).astype(np.float32) + offsets
    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        qnn_bitwidth = 4
        # Replace zeros in scales with 1
        updated_scales = np.where(scales == 0, 1.0, scales)
        qnn_offsets = np.ones(offsets.shape, dtype=np.float32) * (2**(qnn_bitwidth-1)) + offsets / updated_scales
        encodings = generate_encodings(32, qnn_bitwidth, 'INT', 'PER_BLOCK', False, qnn_offsets, updated_scales)
    return values, encodings

def dequantize_q4_0(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1086
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11
    block_size = GGML_BLOCK_SIZES["Q4_0"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # The scales are stored on the first 2 bytes and the rest corresponds to the quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    # scales = np.nan_to_num(scales)
    # the rest of the bytes corresponds to the quants - we discard the first two bytes
    quants = data_u8[:, 2:]

    ql = (quants[:, :] & 0xF).astype(np.int8) - 8
    qr = (quants[:, :] >> 4).astype(np.int8) - 8

    # Use hstack
    quants = np.hstack([ql, qr])
    values = (scales * quants).astype(np.float32)
    qnn_bitwidth = 4
    scales = update_negative_scales_symmetric(values, scales, qnn_bitwidth)
    qnn_offsets = np.zeros(scales.shape, dtype=np.float32)
    encodings = generate_encodings(32, qnn_bitwidth, 'INT', 'PER_BLOCK', True, qnn_offsets, scales)
    return values, encodings


def dequantize_q6_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    # Py implementation
    # https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py#L554
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = len(data) // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    scales = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)

    ql = data_u8[:, :128]
    qh = data_u8[:, 128:192]
    sub_block_scales = data_i8[:, 192:208, np.newaxis].astype(np.float32)

    # Get all sub block scales
    final_scales = (scales * sub_block_scales).reshape((num_blocks, 16, 1))

    # Unpack bits
    ql = ql.reshape((num_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((num_blocks, -1, 32))
    qh = qh.reshape((num_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((num_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((num_blocks, 16, -1)).astype(np.float32)

    # Dequantize
    values = final_scales * q

    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        gguf_bitwidth = 6
        qnn_bitwidth = 8
        qnn_offsets = np.zeros(final_scales.shape, dtype=np.float32)
        # Recompute and update negative scales as per original bw (6)
        final_scales = update_negative_scales_symmetric(values, final_scales, gguf_bitwidth)
        encodings = generate_encodings(16, qnn_bitwidth, 'INT', 'PER_BLOCK', True, qnn_offsets, final_scales)
    return values, encodings


def dequantize_q8_0(data):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    num_blocks = len(data) // block_size

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]
    values = scales * qs

    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        qnn_bitwidth = 8
        qnn_offsets = np.zeros(scales.shape, dtype=np.float32)
        scales = update_negative_scales_symmetric(values, scales, qnn_bitwidth)
        encodings = generate_encodings(32, qnn_bitwidth, 'INT', 'PER_BLOCK', True, qnn_offsets, scales)
    return values, encodings


def dequantize_q2_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q2_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"])

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    factors = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack(
        [
            qs[:, 00:16] >> 0,
            qs[:, 16:32] >> 0,
            qs[:, 00:16] >> 2,
            qs[:, 16:32] >> 2,
            qs[:, 00:16] >> 4,
            qs[:, 16:32] >> 4,
            qs[:, 00:16] >> 6,
            qs[:, 16:32] >> 6,
            qs[:, 32:48] >> 0,
            qs[:, 48:64] >> 0,
            qs[:, 32:48] >> 2,
            qs[:, 48:64] >> 2,
            qs[:, 32:48] >> 4,
            qs[:, 48:64] >> 4,
            qs[:, 32:48] >> 6,
            qs[:, 48:64] >> 6,
            ],
        axis=1,
    )
    scales = d * (factors & 15)
    offsets = dmin * (factors >> 4)
    values = scales * (tmp & 3) - offsets
    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        qnn_bitwidth = 4
        # Replace zeros in scales with 1
        updated_scales = np.where(scales == 0, 1.0, scales)
        qnn_offsets = np.ones(offsets.shape, dtype=np.float32) * (2**(qnn_bitwidth-1)) + (-offsets / updated_scales)
        # As per ggml-quant.c -> Q2_K has only positive scales, so no requirement to update negative scales
        encodings = generate_encodings(16, qnn_bitwidth, 'INT', 'PER_BLOCK', False, qnn_offsets, updated_scales)
    return values, encodings


def dequantize_q3_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q3_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"])

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    # For 256 Quantized weights store 1 bit => 32 bytes
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    # Quantized Weights stores in 64 bytes (2 bits per weight)
    qs = data_u8[:, 32 : 32 + 64].astype(np.int16)
    # 16 Blocks Quantized 6-bit scales
    a, b, c = data_u8[:, 96 : 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    values = (
            d
            * (scales - 32)
            * np.stack(
        [
            (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
            (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
            (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
            (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
            (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
            (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
            (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
            (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
            (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
            (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
            (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
            (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
            (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
            (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
            (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
            (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7]),
        ],
        axis=1,
    )
    )
    scale_factors = d * (scales - 32)
    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        gguf_bitwidth = 3
        qnn_bitwidth = 4
        qnn_offsets = np.zeros(scales.shape, dtype=np.float32)
        # Recompute and update negative scales as per original bw (3)
        scales = update_negative_scales_symmetric(values, scale_factors, gguf_bitwidth)
        encodings = generate_encodings(16, qnn_bitwidth, 'INT', 'PER_BLOCK', True, qnn_offsets, scales)
    return values, encodings


def dequantize_q5_0(data):
    # C implementation
    # https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c#L110
    # C struct definition
    # https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h#L197
    # Py implementation
    # https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py#L316
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q5_0"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_0"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_0"])

    # Extract scales, quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    qh = data_u8[:, 2:6].view(np.uint32)
    ql = data_u8[:, 6:]

    # Unpack bits
    qh = qh.reshape((num_blocks, 1)) >> np.array([bit_idx for bit_idx in range(32)], dtype=np.uint32).reshape((1, 32))
    ql = ql.reshape((num_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh =  (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((num_blocks, -1))

    quants = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)

    # Dequantize
    values = scales * quants.astype(np.float32)

    # generate encodings
    gguf_bitwidth = 5
    qnn_bitwidth = 8
    scales = update_negative_scales_symmetric(values, scales, gguf_bitwidth)
    qnn_offsets = np.zeros(scales.shape, dtype=np.float32)
    encodings = generate_encodings(32, qnn_bitwidth, 'INT', 'PER_BLOCK', True, qnn_offsets, scales)

    return values, encodings


def dequantize_q5_1(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c#L154
    # Py implementation
    # https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py#L359
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q5_1"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_1"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_1"])

    # Extract scales, offsets, quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    offsets = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    qh = data_u8[:, 4:8].view(np.uint32)
    ql = data_u8[:, 8:]

    # Unpack bits
    qh = qh.reshape((num_blocks, 1)) >> np.array([bit_idx for bit_idx in range(32)], dtype=np.uint32).reshape((1, 32))
    ql = ql.reshape((num_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh =  (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((num_blocks, -1))

    quants = (ql | (qh << np.uint8(4))).astype(np.float32)

    # Dequantize
    values = (scales * quants) + offsets

    # generate encodings
    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        qnn_bitwidth = 8
        # Replace zeros in scales with 1
        updated_scales = np.where(scales == 0, 1.0, scales)
        qnn_offsets = np.ones(offsets.shape, dtype=np.float32) * (2**(qnn_bitwidth-1)) + offsets / updated_scales
        encodings = generate_encodings(32, qnn_bitwidth, 'INT', 'PER_BLOCK', False, qnn_offsets, updated_scales)

    return values, encodings


def dequantize_q5_k(data):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    # Py implementation
    # https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py#L527
    num_blocks = len(data) // GGML_BLOCK_SIZES["Q5_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"])

    d = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1).astype(np.float32)
    packed_scales_offsets = data_u8[:, 4:16].reshape(num_blocks, 3, 4)
    qh = data_u8[:, 16 : 16 + 32].reshape(num_blocks, 32)
    qs = data_u8[:, 48 : 48 + 128].reshape(num_blocks, 128)

    # Unpack scales and offsets
    scales_only, offsets_only, rem = np.split(packed_scales_offsets, 3, axis=-2)
    per_block_scales = np.concatenate([scales_only & 0x3F, (rem & 0x0F) | ((scales_only >> 2) & 0x30)], axis=-1)
    per_block_offsets = np.concatenate([offsets_only & 0x3F, (rem >> 4) | ((offsets_only >> 2) & 0x30)], axis=-1)

    per_block_scales = per_block_scales.reshape(num_blocks, 8)
    per_block_offsets = per_block_offsets.reshape(num_blocks, 8)

    final_scales = (d * per_block_scales.astype(np.float32)).reshape(num_blocks, -1, 1)
    final_offsets = (dmin * per_block_offsets.astype(np.float32)).reshape(num_blocks, -1, 1)

    # Get original quant values
    ql = qs.reshape((num_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((num_blocks, -1, 1, 32)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & np.uint8(0x0F)).reshape((num_blocks, -1, 32))
    qh = (qh & np.uint8(0x01)).reshape((num_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.float32)

    values = (final_scales * q - final_offsets)

    encodings = None
    if ENABLE_ASYMMETRIC_MIXED_PRECISION:
        qnn_bitwidth = 8
        updated_scales = np.where(final_scales == 0, 1.0, final_scales)
        qnn_offsets = np.ones(final_scales.shape, dtype=np.float32) * (2**(qnn_bitwidth-1)) + (-final_offsets / updated_scales)
        # As per ggml-quant.c -> Q5_K has only positive scales, so no requirement to update negative scales
        encodings = generate_encodings(32, qnn_bitwidth, 'INT', 'PER_BLOCK', False, qnn_offsets, updated_scales)
    return values, encodings


def load_dequant_gguf_tensor(shape, ggml_type, data):
    if ggml_type in (GGML_TYPES["F32"], GGML_TYPES["F16"]):
        values = data
        encodings = None
    elif ggml_type == GGML_TYPES["Q8_0"]:
        values, encodings = dequantize_q8_0(data)
    elif ggml_type == GGML_TYPES["Q4_0"]:
        values, encodings = dequantize_q4_0(data)
    elif ggml_type == GGML_TYPES["Q4_1"]:
        values, encodings = dequantize_q4_1(data)
    elif ggml_type == GGML_TYPES["Q4_K"]:
        values, encodings = dequantize_q4_k(data)
    elif ggml_type == GGML_TYPES["Q6_K"]:
        values, encodings = dequantize_q6_k(data)
    elif ggml_type == GGML_TYPES["Q2_K"]:
        values, encodings = dequantize_q2_k(data)
    elif ggml_type == GGML_TYPES["Q3_K"]:
        values, encodings = dequantize_q3_k(data)
    elif ggml_type == GGML_TYPES["Q5_0"]:
        values, encodings = dequantize_q5_0(data)
    elif ggml_type == GGML_TYPES["Q5_1"]:
        values, encodings = dequantize_q5_1(data)
    elif ggml_type == GGML_TYPES["Q5_K"]:
        values, encodings = dequantize_q5_k(data)
    else:
        raise NotImplementedError(
            f"ggml_type {ggml_type} not supported"
        )
    # Tensor needs to be reshaped in reverse GGUF shape format as that is the Huggingface Pytorch weight shape.
    # Further explained with the help of an example:
    # Original Pytorch Linear Equation y = x * WT
    # Weight Shape W: O x I,
    # Input Shape  x: N x I,
    # Output Shape y: N x O
    # GGUF serializes the above weight as 1D tensor, but mentions shape as I x O.
    # Thus for reshaping any GGUF 1D tensor into original (pytorch) weight, shape needs to be reversed.
    return values.reshape(shape[::-1]), encodings


class GGUFTokenizerSkeleton:
    def __init__(self, dict_):
        for k, v in dict_.items():
            setattr(self, k, v)

        if not hasattr(self, "merges"):
            if not hasattr(self, "tokens") or not hasattr(self, "scores"):
                raise ValueError(
                    "tokens and scores need to be passed for a LLaMa tokenizer without merges to be instantiated."
                )
            tokens = self.tokens
            scores = self.scores
            vocab = {t: scores[i] for i, t in enumerate(tokens)}

            logger.warning("Merges were not in checkpoint, building merges on the fly.")
            merges = []
            for merge, piece_score in tqdm(vocab.items()):
                local = []
                for index in range(1, len(merge)):
                    piece_l, piece_r = merge[:index], merge[index:]
                    if piece_l in tokens and piece_r in tokens:
                        local.append((piece_l, piece_r, piece_score))
                local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]), reverse=True)
                merges.extend(local)
            merges = sorted(merges, key=lambda val: val[2], reverse=True)
            merges = [(val[0], val[1]) for val in merges]
            self.merges = merges
        else:
            self.merges = [tuple(merge.split(" ")) for merge in self.merges]
            if not hasattr(self, "scores"):
                self.scores = [None for _ in range(len(self.tokens))]

        if not hasattr(self, "added_tokens"):
            self.added_tokens = []

        if not hasattr(self, "unk_token_id"):
            self.unk_token_id = None

        # Llama2 uses the field `unknown_token_id`
        if hasattr(self, "unknown_token_id") and self.unk_token_id is None:
            self.unk_token_id = self.unknown_token_id


# Implementation: https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/convert_slow_tokenizer.py#L106
class GemmaSentencePieceExtractor(SentencePieceExtractor):
    def extract(self, vocab_scores=None) -> Tuple[Dict[str, int], List[Tuple]]:
        """
        By default will return vocab and merges with respect to their order, by sending `vocab_scores` we're going to
        order the merges with respect to the piece scores instead.
        """
        sp = self.sp
        vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())}

        # there is a missing token in the vocab. We have to do this to support merges
        # "<0x09>" is the bytefallback for `\t`
        vocab["\t"] = vocab.get("<0x09>")

        merges = generate_merges(vocab, vocab_scores)
        return vocab, merges


 # Implementation: https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/convert_slow_tokenizer.py#L1277   
class GemmaConverter(SpmConverter):
    handle_byte_fallback = True
    SpmExtractor = GemmaSentencePieceExtractor
    # start and end of turn tokens must be marked as special
    special_tokens = {"<start_of_turn>", "<end_of_turn>"}

    """"
    split_by_unicode_script: true
    split_by_number: true
    split_by_whitespace: true
    treat_whitespace_as_suffix: false
    allow_whitespace_only_pieces: true
    split_digits: true
    byte_fallback: true
    """

    def normalizer(self, proto):
        return normalizers.Replace(" ", "▁")

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.pad_token, 0.0),
            (self.original_tokenizer.eos_token, 0.0),
            (self.original_tokenizer.bos_token, 0.0),
        ]
        for piece in proto.pieces[3:]:
            if piece.piece == "<0x09>":
                vocab += [("\t", piece.score)]
            else:
                vocab += [(piece.piece, piece.score)]
        # vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def pre_tokenizer(self, replacement, add_prefix_space):
        return pre_tokenizers.Split(" ", "merged_with_previous")

    def unk_id(self, proto):
        unk_id = 3
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        return decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )


# Implementation: https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/integrations/ggml.py#L610
class GGUFGemmaConverter(GemmaConverter):
    def __init__(self, tokenizer_dict):
        # set dummy data to avoid unnecessary merges calculation
        tokenizer_dict["merges"] = ["dummy text"]

        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}

    def vocab(self, proto):
        original_vocab = list(zip(proto.tokens, proto.scores))
        updated_vocab = []

        for token, score in original_vocab:
            if token == "<0x09>":
                updated_vocab.append(("\t", score))
            elif " " in token and len(token.strip()) == 0:
                underscores = "▁" * len(token)
                updated_vocab.append((underscores, score))
            else:
                updated_vocab.append((token, score))

        return updated_vocab

    def normalizer(self, proto):
        return normalizers.Replace(" ", "▁")

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def converted(self) -> Tokenizer:
        vocab_scores = self.vocab(self.proto)
        tokenizer = Tokenizer(
            Unigram(
                vocab_scores,
                unk_id=self.proto.unk_token_id,
                byte_fallback=self.handle_byte_fallback,
            )
        )

        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        return tokenizer    


class GGUFLlamaConverter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}
        self.is_llama_3_tokenizer = getattr(self.proto, "tokenizer_type", "llama") != "llama"

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def merges(self, proto):
        return proto.merges

    def tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

        unk_token = proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        bos_token = proto.tokens[proto.bos_token_id] if getattr(proto, "bos_token_id", None) is not None else None
        eos_token = proto.tokens[proto.bos_token_id] if getattr(proto, "eos_token_id", None) is not None else None

        tokenizer = Tokenizer(BPE(bpe_vocab, merges, unk_token=unk_token, fuse_unk=True, byte_fallback=True))

        special_tokens = []

        if not hasattr(self.proto, "token_type"):
            if unk_token is not None:
                special_tokens.append(AddedToken(unk_token, normalized=False, special=True))

            if bos_token is not None:
                special_tokens.append(AddedToken(bos_token, normalized=False, special=True))

            if eos_token is not None:
                special_tokens.append(AddedToken(eos_token, normalized=False, special=True))
        else:
            # 3 stands for special tokens
            special_tokens_idx = np.where(np.array(self.proto.token_type) == 3)[0]

            for idx in special_tokens_idx:
                special_tokens.append(AddedToken(self.proto.tokens[idx], normalized=False, special=True))

        if len(special_tokens) != 0:
            tokenizer.add_special_tokens(special_tokens)

        if len(self.proto.added_tokens) != 0:
            tokenizer.add_tokens(
                [AddedToken(added_token, normalized=False, special=False) for added_token in self.proto.added_tokens]
            )

        self.additional_kwargs["unk_token"] = unk_token
        self.additional_kwargs["eos_token"] = bos_token
        self.additional_kwargs["bos_token"] = eos_token

        if self.is_llama_3_tokenizer:
            self.additional_kwargs["add_prefix_space"] = None
            self.additional_kwargs["clean_up_tokenization_spaces"] = True

            self.additional_kwargs["legacy"] = False
            self.original_tokenizer.legacy = False

        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace("▁", " "),
        ]

        if self.is_llama_3_tokenizer:
            sequence += [decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def converted(self):
        # Copied partly from converted method in SpmConverter class
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        # Prevent the decoder from adding prefix space
        # to avoid including the "Strip" field in the tokenizer.
        tokenizer.decoder = self.decoder(replacement, add_prefix_space=False)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        # HACK: patch the llama-3 tokenizer to use the correspinding pre-tokenizer
        # and normalizer
        if self.is_llama_3_tokenizer:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=False, use_regex=True
            )
            # This is tricky as the additional kwargs are passed after legacy is force-set in LlamaTokenizer's
            # init.
            tokenizer.normalizer = normalizers.Sequence([])

        return tokenizer


class GGUFQwen2Converter(Qwen2Converter):
    def __init__(self, tokenizer_dict):
        self.original_tokenizer = GGUFTokenizerSkeleton(tokenizer_dict)
        self.additional_kwargs = {}

    def converted(self) -> Tokenizer:
        vocab = {word: i for i, word in enumerate(self.original_tokenizer.tokens)}
        merges = self.original_tokenizer.merges
        tokenizer = super().converted(vocab, merges)

        tokenizer.add_special_tokens(
            [
                AddedToken("<|endoftext|>", normalized=False, special=True),
                AddedToken("<|im_start|>", normalized=False, special=True),
                AddedToken("<|im_end|>", normalized=False, special=True),
            ]
        )
        return tokenizer


# Implementation: https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/integrations/ggml.py#L460
class GGUFPhi3Converter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def merges(self, proto):
        return proto.merges

    def tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

        tokenizer = Tokenizer(BPE(bpe_vocab, merges))
        # add the special tokens from phi3 tokenizer config
        tokenizer.add_special_tokens(
            [
                AddedToken("</s>", rstrip=True, lstrip=False, normalized=False, special=True),
                AddedToken("<|endoftext|>", normalized=False, special=True),
                AddedToken("<|assistant|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder1|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder2|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder3|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder4|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|system|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|end|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder5|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder6|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|user|>", rstrip=True, normalized=False, special=True),
            ]
        )

        self.additional_kwargs["unk_token"] = (
            proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        )
        self.additional_kwargs["eos_token"] = (
            proto.tokens[proto.eos_token_id] if proto.eos_token_id is not None else None
        )
        self.additional_kwargs["bos_token"] = (
            proto.tokens[proto.bos_token_id] if proto.bos_token_id is not None else None
        )
        self.additional_kwargs["pad_token"] = (
            proto.tokens[proto.pad_token_id] if proto.pad_token_id is not None else None
        )

        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace(replacement, " "),
        ]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        # Prevent the decoder from adding prefix space
        # to avoid including the "Strip" field in the tokenizer.
        tokenizer.decoder = self.decoder(replacement, add_prefix_space=False)

        return tokenizer


#Implementation: https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/convert_slow_tokenizer.py#L329
class GPT2Converter(Converter):
    def converted(
        self, vocab: Optional[dict[str, int]] = None, merges: Optional[list[tuple[str, str]]] = None
    ) -> Tokenizer:
        if not vocab:
            vocab = self.original_tokenizer.encoder
        if not merges:
            merges = list(self.original_tokenizer.bpe_ranks)

        tokenizer = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        add_prefix_space = getattr(self.original_tokenizer, "add_prefix_space", False)
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        if getattr(self.original_tokenizer, "add_bos_token", False):
            bos = self.original_tokenizer.bos_token
            bos_token_id = self.original_tokenizer.bos_token_id
            tokenizer.post_processor = processors.TemplateProcessing(
                single=f"{bos}:0 $A:0",
                pair=f"{bos}:0 $A:0 $B:1",
                special_tokens=[
                    (bos, bos_token_id),
                ],
            )
        else:
            # XXX trim_offsets=False actually means this post_processor doesn't
            # really do anything.
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        return tokenizer


# Implementation: https://github.com/huggingface/transformers/blob/v4.52.0/src/transformers/integrations/ggml.py#L535
class GGUFGPTConverter(GPT2Converter):
    def __init__(self, tokenizer_dict):
        self.original_tokenizer = GGUFTokenizerSkeleton(tokenizer_dict)
        self.additional_kwargs = {}
        self._is_gpt4o_tokenizer = getattr(self.original_tokenizer, "pre_tokenizer_type", "default") == "gpt-4o"

    def converted(self) -> Tokenizer:
        vocab = {word: i for i, word in enumerate(self.original_tokenizer.tokens)}
        merges = self.original_tokenizer.merges
        tokenizer = super().converted(vocab, merges)

        if self._is_gpt4o_tokenizer:

            # Reference: https://huggingface.co/Xenova/gpt-4o
            pre_tok_regex = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

            pre_tokenizer_list = [pre_tokenizers.Split(Regex(pre_tok_regex), behavior="removed", invert=True),
                                  pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pre_tokenizer_list)

            tokenizer.add_special_tokens(
                [
                    AddedToken("<|endofprompt|>", normalized=False, special=True),
                    AddedToken("<|assistant|>", rstrip=True, normalized=False, special=True),
                    AddedToken("<|end|>", rstrip=True, normalized=False, special=True),
                    AddedToken("<|user|>", rstrip=True, normalized=False, special=True),
                    AddedToken("<|system|>", rstrip=True, normalized=False, special=True),
                    AddedToken("<|tool|>", rstrip=True, normalized=False, special=False),
                    AddedToken("<|/tool|>", rstrip=True, normalized=False, special=False),
                    AddedToken("<|tool_call|>", rstrip=True, normalized=False, special=False),
                    AddedToken("<|/tool_call|>", rstrip=True, normalized=False, special=False),
                    AddedToken("<|tool_response|>", rstrip=True, normalized=False, special=False),
                    AddedToken("<|tag|>", rstrip=True, normalized=False, special=True),
                ]
            )

        return tokenizer


GGUF_TO_FAST_CONVERTERS = {
    "llama": GGUFLlamaConverter,
    "mistral": GGUFLlamaConverter,
    "phi3": GGUFPhi3Converter,
    "qwen2": GGUFQwen2Converter,
    "qwen3": GGUFQwen2Converter,
    "gpt2": GGUFGPTConverter,
    "gemma": GGUFLlamaConverter,
    "gemma2": GGUFGemmaConverter,
    "gemma3": GGUFGemmaConverter,
}


def convert_gguf_tokenizer(architecture, tokenizer_dict) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        architecture (`str`): The model architecture derived from gguf file.
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """
    tokenizer_class_name = architecture
    converter = GGUF_TO_FAST_CONVERTERS[tokenizer_class_name](tokenizer_dict)
    fast_tokenizer = converter.converted()
    return fast_tokenizer, converter.additional_kwargs
