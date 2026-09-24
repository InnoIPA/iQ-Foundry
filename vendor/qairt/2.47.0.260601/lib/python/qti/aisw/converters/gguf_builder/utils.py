# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import re
import onnx
import math
import logging
import itertools
import transformers
import numpy as np

from enum import Enum
from collections import ChainMap
from types import MappingProxyType

from gguf import GGUFReader
from onnx import helper, numpy_helper, TensorProto, ModelProto

logger = logging.getLogger(__name__)

MODEL_TYPE_TO_ARCH = {"llama": "LlamaForCausalLM",
                      "mistral": "MistralForCausalLM",
                      "phi3": "Phi3ForCausalLM",
                      "qwen2": "Qwen2ForCausalLM",
                      "qwen3": "Qwen3ForCausalLM",
                      "gemma": "GemmaForCausalLM",
                      "gemma2": "Gemma2ForCausalLM",
                      "gemma3": "Gemma3ForCausalLM"}

MODEL_TYPE_TO_TOKENIZER = {"llama": transformers.LlamaTokenizerFast,
                           "mistral": transformers.LlamaTokenizerFast,
                           "phi3": transformers.LlamaTokenizerFast,
                           "qwen2": transformers.Qwen2TokenizerFast,
                           "qwen3": transformers.Qwen2TokenizerFast,
                           "gpt2": transformers.GPT2TokenizerFast,
                           "gemma": transformers.GemmaTokenizerFast,
                           "gemma2": transformers.GemmaTokenizerFast,
                           "gemma3": transformers.GemmaTokenizerFast}


SUPPORTED_GGUF_TYPES = {"F32", "F16", "Q2_K", "Q3_K", "Q4_0", "Q4_1", "Q4_K",
                        "Q5_0", "Q5_1", "Q5_K", "Q6_K", "Q8_0"}

GGUF_TO_ONNX_TENSOR_COMMON_MAP = {
    "token_embd": "model.embed_tokens",
    "blk": "model.layers",
    "ffn_up": "mlp.up_proj.{op_type}",
    "ffn_down": "mlp.down_proj.{op_type}",
    "ffn_gate": "mlp.gate_proj.{op_type}",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q": "attn.q_proj.{op_type}",
    "attn_v": "attn.v_proj.{op_type}",
    "attn_k": "attn.k_proj.{op_type}",
    "attn_output": "attn.o_proj.{op_type}",
    "output.weight": "lm_head.{op_type}.weight",
    "output_norm": "model.layers.{max_block}.final_norm_layernorm",
}

GGUF_TO_ONNX_TENSOR = {
    "llama": GGUF_TO_ONNX_TENSOR_COMMON_MAP.copy(),
    "qwen2": GGUF_TO_ONNX_TENSOR_COMMON_MAP.copy(),
    "mistral": GGUF_TO_ONNX_TENSOR_COMMON_MAP.copy(),
    "phi3": GGUF_TO_ONNX_TENSOR_COMMON_MAP.copy(),
    "gemma": GGUF_TO_ONNX_TENSOR_COMMON_MAP.copy(),
    "gemma2": {
        **GGUF_TO_ONNX_TENSOR_COMMON_MAP.copy(),
        "ffn_norm": "pre_feedforward_layernorm",
        "post_ffw_norm": "post_feedforward_layernorm",
        "post_attention_norm": "post_attention_layernorm",
    }
}

ONNX_TENSOR_NAME_STRINGS = {
    "attention_mask_input_name": "attention_mask",
    "position_ids_cos_input_name": "position_ids_cos",
    "position_ids_sin_input_name": "position_ids_sin",
    "llama_final_layernorm": "final_norm_layernorm",
    "llama_SkipLayerNorm": "SkipLayerNorm",
    "llama_LayerNorm": "LayerNorm",
    "llama_GroupQueryAttention": "GroupQueryAttention",
    "llama_qkv_proj": "qkv_proj",
    "llama_name_seqlens_k": "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0",
    "model_attn_mask_node": "/model/attn_mask_reformat/attn_mask_subgraph",
    "model_attn_mask_constant": "/model/constant_nodes/TensorProto.INT64"
}

GGUF_TENSOR_NAME_STRINGS = {
    "embedding_weights_name": "token_embd.weight",
    "lm_head_weights_name": "output.weight"
}

GGUF_TO_MAD_TENSOR_NAME_MAP = {
    "blk": "model.decoder_blocks",
    "weight": "weight",
    "bias": "bias",
    "rope_freqs": "rope_freqs",
    "rope_factors_long": "rope_factors_long",
    "rope_factors_short": "rope_factors_short",
    "token_embd": "model.embedding",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_norm",
    "attn_norm": "pre_norm",
    "attn_q": "q_proj",
    "attn_v": "v_proj",
    "attn_k": "k_proj",
    "attn_output": "o_proj",
    "output": "model.lm_head",
    "output_norm": "model.output_norm",
    "attn_k_norm": "attn_k_norm",
    "attn_q_norm": "attn_q_norm"
}

# Maps GGUF tensor name fragments to their MAD Architecture equivalents.
#
# Structure
# ---------
# The top-level dict has one mandatory key (DEFAULT) plus one optional key per
# architecture that needs overrides:
#
#   DEFAULT  – fallback mappings used by every architecture.  All standard
#              LLaMA-style names belong here.
#
#   "<arch>" – architecture-specific overrides.  Only include keys whose MAD
#              name differs from the DEFAULT.  Lookups first check the
#              architecture section and then fall through to DEFAULT, so you
#              never need to copy unchanged entries.
#
# Unknown architecture fallback
# ------------------------------
# If the requested architecture key is not present in the map at all,
# the lookup automatically returns the DEFAULT section.  This means new
# or unrecognized architectures get sensible LLaMA-style names without
# requiring any code change.
#
# Adding support for a new architecture
# --------------------------------------
# 1. Identify the GGUF tensor name fragments that differ from the DEFAULT
#    mappings in your target architecture's GGUF file.
#
# 2. Add a new key to GGUF_ARCH_TO_MAD_TENSOR_NAME_MAP whose name matches the
#    model_type string used in the config file (e.g. "qwen2").
#    Only list the keys that need a different MAD name:
#
#       "qwen2": {
#           "attn_q": "q_proj_qwen",   # overrides DEFAULT
#           # all other keys fall back to DEFAULT automatically
#       }
#
# 3. Do NOT copy DEFAULT entries into the architecture section – they are
#    inherited automatically via ChainMap after generate_mappings() runs.
#
# 4. Call generate_mappings(GGUF_ARCH_TO_MAD_TENSOR_NAME_MAP) once at module load
#    (already done below).  You do not need to call it again.

DEFAULT = "DEFAULT"
GGUF_ARCH_TO_MAD_TENSOR_NAME_MAP = {
    # Default mappings shared by all architectures (LLaMA-style naming).
    # Keys are GGUF tensor name fragments; values are MAD tensor name fragments.
    DEFAULT : {
        "blk"               : "model.decoder_blocks",
        "weight"            : "weight",
        "bias"              : "bias",
        "rope_freqs"        : "rope_freqs",
        "rope_factors_long" : "rope_factors_long",
        "rope_factors_short": "rope_factors_short",
        "token_embd"        : "model.embedding",
        "ffn_up"            : "mlp.up_proj",
        "ffn_down"          : "mlp.down_proj",
        "ffn_gate"          : "mlp.gate_proj",
        "ffn_norm"          : "post_norm",
        "attn_norm"         : "pre_norm",
        "attn_q"            : "q_proj",
        "attn_v"            : "v_proj",
        "attn_k"            : "k_proj",
        "attn_output"       : "o_proj",
        "output"            : "model.lm_head",
        "output_norm"       : "model.output_norm",
    },
    "qwen3": {
        "attn_k_norm": "k_norm",
        "attn_q_norm": "q_norm",
    },
    "gemma2": {
        "ffn_norm": "pre_ffw_norm",
        "post_ffw_norm": "post_ffw_norm",
        "post_attention_norm": "post_attn_norm",
    },
    "gemma3": {
        "ffn_norm": "pre_ffw_norm",
        "post_ffw_norm": "post_ffw_norm",
        "post_attention_norm": "post_attn_norm",
        "attn_k_norm": "k_norm",
        "attn_q_norm": "q_norm",
    }
}

class _ArchMappingDict(dict):
    """dict that returns the DEFAULT tensor-name section for any unrecognized architecture.

    Python calls __missing__ when __getitem__ cannot find the key, so this works
    transparently through a MappingProxyType wrapper — callers can simply write
    GGUF_ARCH_TO_MAD_TENSOR_NAME_MAP[arch] without any try/except or .get() guard.
    """
    def __missing__(self, key):
        return self[DEFAULT]


def generate_mappings(cfg):
    """Freeze GGUF_ARCH_TO_MAD_TENSOR_NAME_MAP into immutable, chainable views.

    For each architecture key (anything other than DEFAULT) the plain dict is
    replaced with a ChainMap that consults the architecture-specific entries
    first and falls through to DEFAULT for any key not explicitly overridden.
    The entire structure is then wrapped in a MappingProxyType over an
    _ArchMappingDict so that (a) it cannot be mutated at runtime and (b)
    any unknown architecture key automatically returns the DEFAULT section.

    This function mutates *cfg* in place before returning it as an immutable
    proxy; it should only be called once per mapping dict (at module load).
    """
    # Freeze DEFAULT first so the ChainMaps created below share the same
    # immutable proxy rather than holding a reference to the raw dict.
    cfg[DEFAULT] = MappingProxyType(cfg[DEFAULT])

    # iterate over a list so we don't risk issues if structure changes
    for k in list(cfg.keys()):
        if k == DEFAULT:
            continue
        # Each architecture section becomes a two-level ChainMap:
        #   architecture overrides  →  DEFAULT fallback
        cfg[k] = ChainMap(MappingProxyType(cfg[k]), cfg[DEFAULT])

    # _ArchMappingDict.__missing__ returns DEFAULT for any unrecognized arch key.
    return MappingProxyType(_ArchMappingDict(cfg))

GGUF_ARCH_TO_MAD_TENSOR_NAME_MAP = generate_mappings(GGUF_ARCH_TO_MAD_TENSOR_NAME_MAP)


GGUF_TYPE_BLOCK_SIZE = {
    "F32": 1,
    "Q4_0": 32,
    "Q4_1": 32,
    "Q5_0": 32,
    "Q5_1": 32,
    "Q8_0": 32,
    "Q2_K": 16,
    "Q3_K": 16,
    "Q4_K": 32,
    "Q5_K": 32,
    "Q6_K": 16,
}


class GGUFONNXConfig:
    """
    Configuration class for GGUF ONNX conversion.

    Attributes:
        arn_seq_len (int): The sequence length for ARN.
        batch_size (int): The batch size for the model.
        generate_conv_model (bool): Whether to generate a convolutional model.
    """
    def __init__(self, model_config: dict, arn_seq_len: int = 1, batch_size: int = 1, generate_conv_model: bool = False):
        """
        Initializes the GGUFONNXConfig instance.

        Args:
            model_config (dict): The model configuration.
            arn_seq_len (int, optional): The sequence length for ARN. Defaults to 1.
            batch_size (int, optional): The batch size for the model. Defaults to 1.
            generate_conv_model (bool, optional): Whether to generate a convolutional model. Defaults to False.
        """
        self.arn_seq_len = arn_seq_len
        self.batch_size = batch_size
        self.generate_conv_model = generate_conv_model

        # Initialize the relevant model parameters from the model configuration
        self.model_type = model_config["model_type"]
        self.hidden_size = model_config["hidden_size"]
        self.num_heads = model_config["num_attention_heads"]
        self.head_dim = model_config.get('head_dim', self.hidden_size // self.num_heads)
        # Setting the max value of CL as 4k to avoid memory error
        # while serializing the graph (without adaptations) generated by ORT GenAI
        # This is customisable by the user in UL Python API
        self.total_seq_len = min(model_config["max_position_embeddings"], 4096)
        self.num_kv_heads = model_config["num_key_value_heads"] if "num_key_value_heads" in model_config else self.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        # Get the vocabulary size from the model configuration or the model output
        self.vocab_size = model_config.get("vocab_size", None)

    def update(self, **kwargs):
        """
        Updates the configuration attributes.

        Args:
            arn_seq_len (int, optional): The new sequence length for ARN.
            batch_size (int, optional): The new batch size for the model.
            generate_conv_model (bool, optional): Whether to generate a convolutional model.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid attribute: {key}")


def update_symbolic_shape_with_value(model: ModelProto, gguf_onnx_config: GGUFONNXConfig):
    """
        Utility function to update an onnx model inputs, outputs and intermediate value_info shapes.
        All symbolic shapes for the above-mentioned tensors are replaced with constant values.
    """
    total_sequence_length = gguf_onnx_config.total_seq_len
    past_sequence_length = total_sequence_length - gguf_onnx_config.arn_seq_len
    model_input_shape_dict = {
        "batch_size": gguf_onnx_config.batch_size,
        "sequence_length": gguf_onnx_config.arn_seq_len,
        "total_sequence_length": total_sequence_length,
        "past_sequence_length": past_sequence_length
    }

    def update_symbolic_values(value_infos):
        for value_info in value_infos:
            if value_info.type.HasField("tensor_type") and value_info.type.tensor_type.HasField("shape"):
                for cur_dim in value_info.type.tensor_type.shape.dim:
                    if cur_dim.HasField("dim_param") and cur_dim.dim_param in model_input_shape_dict.keys():
                        dict_key = cur_dim.dim_param
                        cur_dim.Clear()
                        cur_dim.dim_value = model_input_shape_dict[dict_key]

    update_symbolic_values(model.graph.input)
    update_symbolic_values(model.graph.output)
    update_symbolic_values(model.graph.value_info)


class LayerNormalization:
    """
    A class to decompose the LayerNormalization operation.

    This class provides methods to decompose layer normalization nodes in an ONNX model into simpler operations.
    """

    def __init__(self, model: onnx.ModelProto, gguf_onnx_config: GGUFONNXConfig):
        """
        Initialize the LayerNormalization class.

        Args:
        model (onnx.ModelProto): The ONNX model to decompose.
        gguf_onnx_config (GGUFONNXConfig): The configuration for the ONNX model.
        """
        self.model = model
        self.hidden_size = gguf_onnx_config.hidden_size

    def __decompose_skip_simplified_layernorm(self):
        """
        Decompose the SkipSimplifiedLayerNormalization nodes in the model into Elementwise Add and SimplifiedLayerNormalization nodes.
        """
        for node in self.model.graph.node:
            if node.op_type == "SkipSimplifiedLayerNormalization":
                # Get input output weight info
                name_sln = node.name
                is_last_layernorm = False
                # Final SkipSLN only has one output
                if ONNX_TENSOR_NAME_STRINGS["llama_final_layernorm"] in name_sln:
                    is_last_layernorm = True
                input_name_data_0 = node.input[0]
                input_name_data_1 = node.input[1]
                weight_name_scale = node.input[2]
                eps_data = node.attribute[0].f
                output_name_data_0 = node.output[0]
                # Set Node Names to be added to graph
                split_name_skip_sln = name_sln.split(ONNX_TENSOR_NAME_STRINGS["llama_SkipLayerNorm"])
                node_name_elementwise_add = split_name_skip_sln[0] + "elementwise_add"
                node_name_simplifiedlayernorm = split_name_skip_sln[0] + "LayerNorm"
                if not is_last_layernorm:
                    output_name_data_3 = node.output[3]
                    node_elementwise_add = helper.make_node("Add", name=node_name_elementwise_add,
                                                            inputs=[input_name_data_0, input_name_data_1],
                                                            outputs=[output_name_data_3])
                    node_simplifiedlayernorm = helper.make_node("SimplifiedLayerNormalization",
                                                                name=node_name_simplifiedlayernorm,
                                                                inputs=[output_name_data_3, weight_name_scale],
                                                                outputs=[output_name_data_0])
                else:
                    output_name_elemwise_add = node_name_elementwise_add + "/output_0"
                    node_elementwise_add = helper.make_node("Add", name=node_name_elementwise_add,
                                                            inputs=[input_name_data_0, input_name_data_1],
                                                            outputs=[output_name_elemwise_add])
                    node_simplifiedlayernorm = helper.make_node("SimplifiedLayerNormalization",
                                                                name=node_name_simplifiedlayernorm,
                                                                inputs=[output_name_elemwise_add, weight_name_scale],
                                                                outputs=[output_name_data_0])
                    elemwise_add_vi = helper.make_tensor_value_info(output_name_elemwise_add, TensorProto.FLOAT,
                                                                    ["batch_size", "sequence_length", self.hidden_size])
                    self.model.graph.value_info.extend([elemwise_add_vi])
                # Add required attributes to SimplfiedLayerNorm
                eps_attribute = helper.make_attribute("epsilon", eps_data)
                axis_attribute = helper.make_attribute("axis", -1)
                node_simplifiedlayernorm.attribute.extend([eps_attribute, axis_attribute])
                self.model.graph.node.extend([node_elementwise_add, node_simplifiedlayernorm])
                # Remove Node
                self.model.graph.node.remove(node)

    def __decompose_simplified_layernorm(self):
        """
        Decompose the SimplifiedLayerNormalization nodes in the model into constituent ONNX ops.
        """
        for node in self.model.graph.node:
            if node.op_type == 'SimplifiedLayerNormalization':
                # Get input output weight info
                name_sln = node.name
                input_name_data = node.input[0]
                weight_name_scale = node.input[1]
                eps_data = node.attribute[0].f
                output_name_data = node.output[0]
                # Set Node Names to be added to graph
                split_name_sln = name_sln.split(ONNX_TENSOR_NAME_STRINGS["llama_LayerNorm"])
                node_name_pow_value = split_name_sln[0] + 'pow_value'
                node_name_eps_value = split_name_sln[0] + 'eps_value'
                node_name_square_inp = split_name_sln[0] + 'square_inp'
                node_name_reduce_mean = split_name_sln[0] + 'reduce_mean'
                node_name_add_eps = split_name_sln[0] + 'add_eps'
                node_name_sqrt = split_name_sln[0] + 'sqrt'
                node_name_elementwise_div = split_name_sln[0] + 'elementwise_div'
                node_name_elementwise_mul_gamma = split_name_sln[0] + 'elementwise_mul_gamma'
                # Set constants Input
                pow_value = np.array([2], dtype=np.float32)
                eps_value = np.array([eps_data], dtype=np.float32)
                # Set Output names for nodes
                out_name_pow_value = node_name_pow_value + '/output_0'
                out_name_eps_value = node_name_eps_value + '/output_0'
                out_name_square_inp = node_name_square_inp + '/output_0'
                out_name_reduce_mean = node_name_reduce_mean + '/output_0'
                out_name_add_eps = node_name_add_eps + '/output_0'
                out_name_sqrt = node_name_sqrt + '/output_0'
                out_name_elementwise_div = node_name_elementwise_div + '/output_0'
                # Create Decomposed RMSNorm Nodes
                node_constant_pow_value = helper.make_node('Constant', inputs=[], outputs=[out_name_pow_value],
                                                           value=helper.make_tensor(
                                                               name=node_name_pow_value,
                                                               data_type=helper.np_dtype_to_tensor_dtype(np.dtype('float32')),
                                                               dims=pow_value.shape,
                                                               vals=pow_value.flatten()))
                node_constant_eps_value = helper.make_node('Constant', inputs=[], outputs=[out_name_eps_value],
                                                           value=helper.make_tensor(
                                                               name=node_name_eps_value,
                                                               data_type=helper.np_dtype_to_tensor_dtype(np.dtype('float32')),
                                                               dims=eps_value.shape,
                                                               vals=eps_value.flatten()))
                node_square_inp = helper.make_node('Pow', name=node_name_square_inp,
                                                   inputs=[input_name_data, out_name_pow_value],
                                                   outputs=[out_name_square_inp])
                node_reduce_mean = helper.make_node('ReduceMean', name=node_name_reduce_mean, inputs=[out_name_square_inp],
                                                    axes=[2],
                                                    outputs=[out_name_reduce_mean])
                node_add_eps = helper.make_node('Add', name=node_name_add_eps,
                                                inputs=[out_name_reduce_mean, out_name_eps_value],
                                                outputs=[out_name_add_eps])
                node_sqrt = helper.make_node('Sqrt', name=node_name_sqrt, inputs=[out_name_add_eps],
                                             outputs=[out_name_sqrt])
                node_elementwise_div = helper.make_node('Div', name=node_name_elementwise_div,
                                                        inputs=[input_name_data, out_name_sqrt],
                                                        outputs=[out_name_elementwise_div])
                node_elementwise_mul_gamma = helper.make_node('Mul', name=node_name_elementwise_mul_gamma,
                                                              inputs=[out_name_elementwise_div, weight_name_scale],
                                                              outputs=[output_name_data])
                # Create intermediate output tensors and add to graph: Value_Info
                vi_square_inp = helper.make_tensor_value_info(out_name_square_inp, TensorProto.FLOAT,
                                                                  ['batch_size', 'sequence_length', self.hidden_size])
                vi_reduce_mean = helper.make_tensor_value_info(out_name_reduce_mean, TensorProto.FLOAT,
                                                               ['batch_size', 'sequence_length', 1])
                vi_add_eps = helper.make_tensor_value_info(out_name_add_eps, TensorProto.FLOAT,
                                                           ['batch_size', 'sequence_length', 1])
                vi_sqrt = helper.make_tensor_value_info(out_name_sqrt, TensorProto.FLOAT,
                                                        ['batch_size', 'sequence_length', 1])
                vi_elementwise_div = helper.make_tensor_value_info(out_name_elementwise_div, TensorProto.FLOAT,
                                                                   ['batch_size', 'sequence_length', self.hidden_size])
                self.model.graph.value_info.extend([vi_square_inp, vi_reduce_mean, vi_add_eps,
                                           vi_sqrt, vi_elementwise_div])
                # Add created nodes to graph
                self.model.graph.node.extend([node_constant_pow_value, node_constant_eps_value,
                                     node_square_inp, node_reduce_mean, node_add_eps, node_sqrt,
                                     node_elementwise_div, node_elementwise_mul_gamma])
                # Remove Node
                self.model.graph.node.remove(node)

    def decompose(self):
        """
        Decompose the layer norm ops in the model with constituent ONNX ops.
        """
        self.__decompose_skip_simplified_layernorm()
        self.__decompose_simplified_layernorm()


# Define a class to encapsulate the model input and output updating functionality
class ModelInputOutputUpdater:
    """
    Class responsible for updating the model inputs and outputs.
    """

    # Initialize the ModelInputOutputUpdater instance with the model, model configuration, and GGUF ONNX configuration
    def __init__(self, model: ModelProto, gguf_onnx_config: GGUFONNXConfig):
        """
        Initializes the ModelInputOutputUpdater instance.
        Args:
            model (ModelProto): The model to be updated.
            gguf_onnx_config (GGUFONNXConfig): The GGUF ONNX configuration.
        """
        # Store the model, model configuration, and GGUF ONNX configuration as instance variables
        self.model = model
        # Initialize the sequence length and batch size from the GGUF ONNX configuration
        self.cur_seq_len = gguf_onnx_config.arn_seq_len
        self.batch_size = gguf_onnx_config.batch_size

        # Initialize the hidden size, number of attention heads, and head dimension from the model configuration
        self.hidden_size = gguf_onnx_config.hidden_size
        self.num_heads = gguf_onnx_config.num_heads
        self.head_dim = gguf_onnx_config.head_dim
        self.total_seq_len = gguf_onnx_config.total_seq_len
        self.num_kv_heads = gguf_onnx_config.num_kv_heads
        self.n_rep = gguf_onnx_config.n_rep

        # Define the name of the attention mask input
        self.input_name_attn_mask = ONNX_TENSOR_NAME_STRINGS["attention_mask_input_name"]
        # Define the names of the position IDs inputs
        self.pos_ids_cos_name = ONNX_TENSOR_NAME_STRINGS["position_ids_cos_input_name"]
        self.pos_ids_sin_name = ONNX_TENSOR_NAME_STRINGS["position_ids_sin_input_name"]

    # Method to update the input data types of the model
    def update_input_datatypes(self):
        """
        Updates the input_ids datatype of the model to int32.
        """
        # Iterate over the model's input values
        for vi in self.model.graph.input:
            # Check if the input value is named "input_ids"
            if vi.name == "input_ids":
                # Update the data type of the input value to int32
                vi.type.tensor_type.elem_type = 6

    # Method to update the attention mask of the model
    def update_attention_mask(self):
        """
        Updates the attention mask of the model.
        """
        # Remove the existing attention mask from the model's input
        for model_inp in self.model.graph.input:
            if model_inp.name == self.input_name_attn_mask:
                self.model.graph.input.remove(model_inp)

        # Define the node types to remove from the model's graph
        attn_mask_branch_nodes_remove = ["ReduceSum", "Sub", "Cast", "Shape", "Gather"]

        # Initialize a list to store the nodes to remove
        attn_nodes_list = []

        # Iterate over the model's graph nodes
        for node in self.model.graph.node:
            # Check if the node is related to the attention mask
            if ONNX_TENSOR_NAME_STRINGS["model_attn_mask_node"] in node.name and any(
                    [node_type for node_type in attn_mask_branch_nodes_remove if node.op_type == node_type]):
                # Add the node to the list of nodes to remove
                attn_nodes_list.append(node)
            # Check if the node is a constant related to the attention mask
            elif ONNX_TENSOR_NAME_STRINGS["model_attn_mask_constant"] in node.name and node.op_type == "Constant":
                attn_nodes_list.append(node)

        # Remove the nodes from the model's graph
        for node in attn_nodes_list:
            self.model.graph.node.remove(node)

        # Remove the value information related to the attention mask from the model's graph
        for vi in self.model.graph.value_info:
            if ONNX_TENSOR_NAME_STRINGS["model_attn_mask_node"] in vi.name:
                self.model.graph.value_info.remove(vi)

        # Define the shape of the new attention mask
        attn_mask_new_shape = [self.batch_size, 1, self.cur_seq_len, self.total_seq_len]

        # Create a new attention mask input
        new_attn_mask_input = helper.make_tensor_value_info(self.input_name_attn_mask, TensorProto.FLOAT, attn_mask_new_shape)

        # Insert the new attention mask input into the model's graph
        self.model.graph.input.insert(1, new_attn_mask_input)

    # Method to update the model inputs for RoPE
    def update_model_inputs_rope(self):
        """
        Updates the model inputs for RoPE.
        """
        # Define the shapes of the position IDs inputs
        pos_ids_cos_shape = [self.batch_size, 1, self.cur_seq_len, self.head_dim // 2]
        pos_ids_sin_shape = [self.batch_size, 1, self.cur_seq_len, self.head_dim // 2]

        # Create the position IDs inputs
        pos_ids_cos_input = helper.make_tensor_value_info(self.pos_ids_cos_name, TensorProto.FLOAT, pos_ids_cos_shape)
        pos_ids_sin_input = helper.make_tensor_value_info(self.pos_ids_sin_name, TensorProto.FLOAT, pos_ids_sin_shape)

        # Insert the position IDs inputs into the model's graph after input_ids(0) and attention_mask(1)
        self.model.graph.input.insert(2, pos_ids_cos_input)
        self.model.graph.input.insert(3, pos_ids_sin_input)

        # Remove the sin and cos caches from the model's initializers
        for init in self.model.graph.initializer:
            if init.name == "sin_cache" or init.name == "cos_cache":
                self.model.graph.initializer.remove(init)

    # Method to update the KV cache input and output names
    def update_kv_cache_input_output_names(self):
        """
        Updates the KV cache input and output names.
        """
        # Iterate over the model's input values (excluding the first two)
        for vi in itertools.islice(self.model.graph.input, 2, None):
            # Get the name of the input value
            vi_name = vi.name
            # Split the name into parts
            vi_name_parts = vi_name.split(".")
            # Create a new name for the input value
            new_vi_name = f"past_{vi_name_parts[-1]}_{vi_name_parts[-2]}_in"

            # Define the new dimensions of the input based on whether it is past_key or past_value data
            new_vi_dims = [self.batch_size, self.num_kv_heads, self.head_dim, self.total_seq_len - self.cur_seq_len] if vi_name_parts[
                                                                                                           -1] == "key" else [
                self.batch_size, self.num_kv_heads, self.total_seq_len - self.cur_seq_len, self.head_dim]

            # Update the name and dimensions of the input
            vi.name = new_vi_name
            for idx, dim in enumerate(vi.type.tensor_type.shape.dim):
                dim.dim_value = new_vi_dims[idx]

            # Update the input names of the GroupQueryAttention nodes
            for node in self.model.graph.node:
                if node.op_type == "GroupQueryAttention":
                    for idx, node_input in enumerate(node.input):
                        if node_input == vi_name:
                            node.input[idx] = new_vi_name

        # Iterate over the model's output values (excluding the first one)
        for vi in itertools.islice(self.model.graph.output, 1, None):
            # Check if the output value is related to the present KV cache
            if "present" in vi.name:
                # Get the name of the output value
                vi_name = vi.name
                # Split the name into parts
                vi_name_parts = vi_name.split(".")
                # Create a new name for the output value
                new_vi_name = f"past_{vi_name_parts[-1]}_{vi_name_parts[-2]}_out"

                # Define the new dimensions of the output value
                new_vi_dims = [self.batch_size, self.num_kv_heads, self.head_dim, self.cur_seq_len] if vi_name_parts[-1] == "key" else [
                    self.batch_size, self.num_kv_heads, self.cur_seq_len, self.head_dim]

                # Update the name and dimensions of the output value
                vi.name = new_vi_name
                for idx, dim in enumerate(vi.type.tensor_type.shape.dim):
                    dim.dim_value = new_vi_dims[idx]

                # Update the output names of the GroupQueryAttention nodes
                for node in self.model.graph.node:
                    if node.op_type == "GroupQueryAttention":
                        for idx, node_input in enumerate(node.output):
                            if node_input == vi_name:
                                node.output[idx] = new_vi_name


class GroupQueryAttention:
    """
    GroupQueryAttention (GQA) Utility Class to decompose ort-genai generated GQA Op into constituent ONNX Ops.
    """

    def __init__(self, model: ModelProto, gguf_onnx_config: GGUFONNXConfig):
        self.model = model
        self.hidden_size = gguf_onnx_config.hidden_size
        self.num_heads = gguf_onnx_config.num_heads
        self.total_seq_len = gguf_onnx_config.total_seq_len
        self.num_kv_heads = gguf_onnx_config.num_kv_heads
        self.head_dim = gguf_onnx_config.head_dim
        self.n_rep = gguf_onnx_config.n_rep
        self.cur_seq_len = gguf_onnx_config.arn_seq_len
        self.batch_size = gguf_onnx_config.batch_size
        self.generate_conv_model = gguf_onnx_config.generate_conv_model
        self.input_name_seqlens_k = ONNX_TENSOR_NAME_STRINGS["llama_name_seqlens_k"]
        # Make attn_mask_name attribute of class
        self.input_name_attn_mask = ONNX_TENSOR_NAME_STRINGS["attention_mask_input_name"]
        self.pos_ids_cos_name = ONNX_TENSOR_NAME_STRINGS["position_ids_cos_input_name"]
        self.pos_ids_sin_name = ONNX_TENSOR_NAME_STRINGS["position_ids_sin_input_name"]

    def __create_rope(self, tensor_type: str, split_name: str, input_tensor_name: str, pos_ids_sin_name: str,
                      pos_ids_cos_name: str):

        # Set Node Names to be added to graph. tensor_type = "q" OR "k"
        rope_num_heads = None
        if tensor_type == "q":
            rope_num_heads = self.num_heads
        elif tensor_type == "k":
            rope_num_heads = self.num_kv_heads

        node_name_slice_t1 = split_name + "slice_" + tensor_type + "1"
        node_name_slice_t2 = split_name + "slice_" + tensor_type + "2"
        node_name_mul_t1_sin = split_name + "mul_" + tensor_type + "1_sin"
        node_name_mul_t1_cos = split_name + "mul_" + tensor_type + "1_cos"
        node_name_mul_t2_sin = split_name + "mul_" + tensor_type + "2_sin"
        node_name_mul_t2_cos = split_name + "mul_" + tensor_type + "2_cos"
        node_name_add_t1_sin_t2_cos = split_name + "add_" + tensor_type + "1_sin_" + tensor_type + "2_cos"
        node_name_sub_t1_cos_t2_sin = split_name + "sub_" + tensor_type + "1_cos_" + tensor_type + "2_sin"
        node_name_concat_t1_t2 = split_name + "concat_" + tensor_type + "1_" + tensor_type + "2"

        # Set Output names for nodes
        out_name_slice_t1 = node_name_slice_t1 + '/output_0'
        out_name_slice_t2 = node_name_slice_t2 + '/output_0'
        out_name_mul_t1_sin = node_name_mul_t1_sin + '/output_0'
        out_name_mul_t1_cos = node_name_mul_t1_cos + '/output_0'
        out_name_mul_t2_sin = node_name_mul_t2_sin + '/output_0'
        out_name_mul_t2_cos = node_name_mul_t2_cos + '/output_0'
        out_name_add_t1_sin_t2_cos = node_name_add_t1_sin_t2_cos + '/output_0'
        out_name_sub_t1_cos_t2_sin = node_name_sub_t1_cos_t2_sin + '/output_0'
        out_name_rope_t = node_name_concat_t1_t2 + '/output_0'

        # RoPE Nodes Q
        node_name_constant_slice = split_name + tensor_type + '_slice'
        node_name_start_1 = node_name_constant_slice + '/start_1'
        node_name_start_2 = node_name_constant_slice + '/start_2'
        node_name_end_1 = node_name_constant_slice + '/end_1'
        node_name_end_2 = node_name_constant_slice + '/end_2'
        node_name_slice_axes = node_name_constant_slice + '/axes'
        node_name_slice_steps = node_name_constant_slice + '/steps'

        out_name_start_1 = node_name_start_1 + '/output_0'
        out_name_start_2 = node_name_start_2 + '/output_0'
        out_name_end_1 = node_name_end_1 + '/output_0'
        out_name_end_2 = node_name_end_2 + '/output_0'
        out_name_slice_axes = node_name_slice_axes + '/output_0'
        out_name_slice_steps = node_name_slice_steps + '/output_0'

        node_constant_start_1 = helper.make_node('Constant', inputs=[], outputs=[out_name_start_1],
                                                 value=helper.make_tensor(
                                                     name=node_name_start_1,
                                                     data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                     dims=[1],
                                                     vals=[0]))
        node_constant_end_1 = helper.make_node('Constant', inputs=[], outputs=[out_name_end_1],
                                               value=helper.make_tensor(
                                                   name=node_name_start_1,
                                                   data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                   dims=[1],
                                                   vals=[self.head_dim // 2]))
        node_constant_start_2 = helper.make_node('Constant', inputs=[], outputs=[out_name_start_2],
                                                 value=helper.make_tensor(
                                                     name=node_name_start_2,
                                                     data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                     dims=[1],
                                                     vals=[self.head_dim // 2]))
        node_constant_end_2 = helper.make_node('Constant', inputs=[], outputs=[out_name_end_2],
                                               value=helper.make_tensor(
                                                   name=node_name_start_2,
                                                   data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                   dims=[1],
                                                   vals=[self.head_dim]))
        node_constant_slice_axes = helper.make_node('Constant', inputs=[], outputs=[out_name_slice_axes],
                                                    value=helper.make_tensor(
                                                        name=node_name_slice_axes,
                                                        data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                        dims=[1],
                                                        vals=[-1]))
        node_constant_slice_steps = helper.make_node('Constant', inputs=[], outputs=[out_name_slice_steps],
                                                    value=helper.make_tensor(
                                                        name=node_name_slice_steps,
                                                        data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                        dims=[1],
                                                        vals=[1]))
        node_slice_t1 = helper.make_node('Slice', name=node_name_slice_t1,
                                         inputs=[input_tensor_name,
                                                 out_name_start_1,
                                                 out_name_end_1,
                                                 out_name_slice_axes,
                                                 out_name_slice_steps],
                                         outputs=[out_name_slice_t1])
        node_slice_t2 = helper.make_node('Slice', name=node_name_slice_t2,
                                         inputs=[input_tensor_name,
                                                 out_name_start_2,
                                                 out_name_end_2,
                                                 out_name_slice_axes,
                                                 out_name_slice_steps],
                                         outputs=[out_name_slice_t2])
        node_mul_t1_sin = helper.make_node('Mul', name=node_name_mul_t1_sin,
                                           inputs=[out_name_slice_t1, pos_ids_sin_name],
                                           outputs=[out_name_mul_t1_sin])
        node_mul_t1_cos = helper.make_node('Mul', name=node_name_mul_t1_cos,
                                           inputs=[out_name_slice_t1, pos_ids_cos_name],
                                           outputs=[out_name_mul_t1_cos])
        node_mul_t2_sin = helper.make_node('Mul', name=node_name_mul_t2_sin,
                                           inputs=[out_name_slice_t2, pos_ids_sin_name],
                                           outputs=[out_name_mul_t2_sin])
        node_mul_t2_cos = helper.make_node('Mul', name=node_name_mul_t2_cos,
                                           inputs=[out_name_slice_t2, pos_ids_cos_name],
                                           outputs=[out_name_mul_t2_cos])
        node_add_t1_sin_t2_cos = helper.make_node('Add', name=node_name_add_t1_sin_t2_cos,
                                                  inputs=[out_name_mul_t1_sin, out_name_mul_t2_cos],
                                                  outputs=[out_name_add_t1_sin_t2_cos])
        node_sub_t1_cos_t2_sin = helper.make_node('Sub', name=node_name_sub_t1_cos_t2_sin,
                                                  inputs=[out_name_mul_t1_cos, out_name_mul_t2_sin],
                                                  outputs=[out_name_sub_t1_cos_t2_sin])
        node_concat_t1_t2 = helper.make_node('Concat', name=node_name_concat_t1_t2,
                                             inputs=[out_name_sub_t1_cos_t2_sin, out_name_add_t1_sin_t2_cos],
                                             axis=-1, outputs=[out_name_rope_t])

        # Create intermediate output tensors and add to graph: Value_Info
        vi_slice_t1 = helper.make_tensor_value_info(out_name_slice_t1, TensorProto.FLOAT,
                                                    [self.batch_size, rope_num_heads, 'sequence_length',
                                                     self.head_dim // 2])
        vi_slice_t2 = helper.make_tensor_value_info(out_name_slice_t2, TensorProto.FLOAT,
                                                    [self.batch_size, rope_num_heads, 'sequence_length',
                                                     self.head_dim // 2])
        vi_mul_t1_sin = helper.make_tensor_value_info(out_name_mul_t1_sin, TensorProto.FLOAT,
                                                      [self.batch_size, rope_num_heads, 'sequence_length',
                                                       self.head_dim // 2])
        vi_mul_t1_cos = helper.make_tensor_value_info(out_name_mul_t1_cos, TensorProto.FLOAT,
                                                      [self.batch_size, rope_num_heads, 'sequence_length',
                                                       self.head_dim // 2])
        vi_mul_t2_sin = helper.make_tensor_value_info(out_name_mul_t2_sin, TensorProto.FLOAT,
                                                      [self.batch_size, rope_num_heads, 'sequence_length',
                                                       self.head_dim // 2])
        vi_mul_t2_cos = helper.make_tensor_value_info(out_name_mul_t2_cos, TensorProto.FLOAT,
                                                      [self.batch_size, rope_num_heads, 'sequence_length',
                                                       self.head_dim // 2])
        vi_add_t1_sin_t2_cos = helper.make_tensor_value_info(out_name_add_t1_sin_t2_cos, TensorProto.FLOAT,
                                                             [self.batch_size, rope_num_heads, 'sequence_length',
                                                              self.head_dim // 2])
        vi_sub_t1_cos_t2_sin = helper.make_tensor_value_info(out_name_sub_t1_cos_t2_sin, TensorProto.FLOAT,
                                                             [self.batch_size, rope_num_heads, 'sequence_length',
                                                              self.head_dim // 2])
        vi_rope_t = helper.make_tensor_value_info(out_name_rope_t, TensorProto.FLOAT,
                                                  [self.batch_size, rope_num_heads, 'sequence_length', self.head_dim])

        # Add new value-Infos to graph
        self.model.graph.value_info.extend([vi_slice_t1, vi_slice_t2, vi_mul_t1_sin,
                                            vi_mul_t1_cos, vi_mul_t2_sin, vi_mul_t2_cos,
                                            vi_add_t1_sin_t2_cos, vi_sub_t1_cos_t2_sin, vi_rope_t])

        # Add created nodes to graph
        self.model.graph.node.extend([node_constant_start_1, node_constant_start_2,
                                      node_constant_end_1, node_constant_end_2,
                                      node_constant_slice_steps, node_constant_slice_axes,
                                      node_slice_t1, node_slice_t2, node_mul_t1_sin, node_mul_t1_cos,
                                      node_mul_t2_sin, node_mul_t2_cos,
                                      node_add_t1_sin_t2_cos, node_sub_t1_cos_t2_sin, node_concat_t1_t2])
        return out_name_rope_t

    def __add_repetition_nodes(self, tensor_type: str, split_name: str, input_name_past_tensor: str,
                               input_name_present_tensor: str, initializer_name_expand_tensor: str):
        """Add nodes for key / value repetition in GQA when num_kv_heads != num_heads """

        # tensor_type = "k" OR "v"
        shape_concat_t = None
        shape_current_t = None
        axis_dim = None
        if tensor_type == "k":
            shape_concat_t = np.array([self.batch_size, self.num_kv_heads, 1, self.head_dim, self.total_seq_len], dtype=np.int64)
            shape_current_t = np.array([self.batch_size, self.num_kv_heads * self.n_rep, self.head_dim, self.total_seq_len],
                                       dtype=np.int64)
            axis_dim = -1
        elif tensor_type == "v":
            shape_concat_t = np.array([self.batch_size, self.num_kv_heads, 1, self.total_seq_len, self.head_dim], dtype=np.int64)
            shape_current_t = np.array([self.batch_size, self.num_kv_heads * self.n_rep, self.total_seq_len, self.head_dim],
                                       dtype=np.int64)
            axis_dim = -2

        # Set Node Names to be added to graph
        node_name_concat_t = split_name + tensor_type + "_concat"
        node_name_shape_concat_t = split_name + "shape_concat_" + tensor_type
        node_name_shape_current_t = split_name + "shape_current_" + tensor_type
        node_name_reshape_concat_t = split_name + "reshape_concat_" + tensor_type
        node_name_expand_t = split_name + "expand_" + tensor_type
        node_name_reshape_current_t = split_name + "current_" + tensor_type + "_reshape"

        # Set Output names for nodes
        out_name_concat_t = node_name_concat_t + '/output_0'
        out_name_shape_concat_t = node_name_shape_concat_t + '/output_0'
        out_name_shape_current_t = node_name_shape_current_t + '/output_0'
        out_name_reshape_concat_t = node_name_reshape_concat_t + '/output_0'
        out_name_expand_t = node_name_expand_t + '/output_0'
        out_name_reshape_current_t = node_name_reshape_current_t + '/output_0'

        # Concat Nodes
        node_concat_t = helper.make_node('Concat', name=node_name_concat_t,
                                         inputs=[input_name_past_tensor, input_name_present_tensor],
                                         axis=axis_dim, outputs=[out_name_concat_t])
        # Tensor Repetition Nodes
        node_constant_shape_concat_t = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_concat_t],
                                                        value=helper.make_tensor(
                                                            name=node_name_shape_concat_t,
                                                            data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                            dims=shape_concat_t.shape,
                                                            vals=shape_concat_t.flatten().tolist()))
        node_constant_shape_current_t = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_current_t],
                                                         value=helper.make_tensor(
                                                             name=node_name_shape_current_t,
                                                             data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                             dims=shape_current_t.shape,
                                                             vals=shape_current_t.flatten()))
        node_reshape_concat_t = helper.make_node('Reshape', name=node_name_reshape_concat_t, inputs=[out_name_concat_t, out_name_shape_concat_t],
                                                 outputs=[out_name_reshape_concat_t])
        node_expand_t = helper.make_node('Mul', name=node_name_expand_t, inputs=[out_name_reshape_concat_t, initializer_name_expand_tensor],
                                         outputs=[out_name_expand_t])
        node_reshape_current_t = helper.make_node('Reshape', name=node_name_reshape_current_t, inputs=[out_name_expand_t, out_name_shape_current_t],
                                                  outputs=[out_name_reshape_current_t])

        # Create intermediate output tensors and add to graph: Value_Info
        vi_concat_t = helper.make_tensor_value_info(out_name_concat_t, TensorProto.FLOAT,
                                                        [self.batch_size, self.num_kv_heads, shape_concat_t[-2].item(), shape_concat_t[-1].item()])
        vi_reshape_concat_t = helper.make_tensor_value_info(out_name_reshape_concat_t, TensorProto.FLOAT,
                                                            shape_concat_t.tolist())
        vi_expand_t = helper.make_tensor_value_info(out_name_expand_t, TensorProto.FLOAT,
                                                    [self.batch_size, self.num_kv_heads, self.n_rep, shape_concat_t[-2].item(), shape_concat_t[-1].item()])
        vi_reshape_current_t = helper.make_tensor_value_info(out_name_reshape_current_t, TensorProto.FLOAT,
                                                             shape_current_t.tolist())

        self.model.graph.value_info.extend([vi_concat_t, vi_reshape_concat_t, vi_expand_t, vi_reshape_current_t])

        # Add created nodes to graph
        self.model.graph.node.extend([node_concat_t, node_constant_shape_concat_t, node_constant_shape_current_t,
                                      node_reshape_concat_t, node_expand_t, node_reshape_current_t])

        return out_name_reshape_current_t

    def __scaled_dot_product_attention(self, split_name: str, input_q: str, input_k_transpose: str, input_v: str):
        div_value = np.array([math.sqrt(self.head_dim)], dtype=np.float32)

        node_name_div_value = split_name + 'div_value'
        node_name_matmul_qk = split_name + 'matmul_qk'
        node_name_add_qk_attn_mask = split_name + 'add_qk_attn_mask'
        node_name_div_qk = split_name + 'div_qk'
        node_name_softmax_qk = split_name + 'softmax_qk'
        node_name_matmul_attnv = split_name + 'matmul_attnv'

        out_name_div_value = node_name_div_value + '/output_0'
        out_name_matmul_qk = node_name_matmul_qk + '/output_0'
        out_name_add_qk_attn_mask = node_name_add_qk_attn_mask + '/output_0'
        out_name_div_qk = node_name_div_qk + '/output_0'
        out_name_softmax_qk = node_name_softmax_qk + '/output_0'
        out_name_matmul_attnv = node_name_matmul_attnv + '/output_0'

        # Q * K'
        node_matmul_qk = helper.make_node('MatMul', name=node_name_matmul_qk, inputs=[input_q, input_k_transpose],
                                          outputs=[out_name_matmul_qk])
        # Constant Div
        node_constant_div_qk = helper.make_node('Constant', inputs=[], outputs=[out_name_div_value],
                                                value=helper.make_tensor(
                                                    name=node_name_div_value,
                                                    data_type=helper.np_dtype_to_tensor_dtype(np.dtype('float32')),
                                                    dims=div_value.shape,
                                                    vals=div_value.flatten()))
        node_div_qk = helper.make_node('Div', name=node_name_div_qk, inputs=[out_name_matmul_qk, out_name_div_value],
                                       outputs=[out_name_div_qk])
        node_add_qk_attn_mask = helper.make_node('Add', name=node_name_add_qk_attn_mask,
                                                 inputs=[out_name_div_qk, self.input_name_attn_mask],
                                                 outputs=[out_name_add_qk_attn_mask])
        # Softmax Node
        node_softmax_qk = helper.make_node('Softmax', name=node_name_softmax_qk, inputs=[out_name_add_qk_attn_mask],
                                           outputs=[out_name_softmax_qk])
        # Attn * V
        node_matmul_attnv = helper.make_node('MatMul', name=node_name_matmul_attnv,
                                             inputs=[out_name_softmax_qk, input_v],
                                             outputs=[out_name_matmul_attnv])

        # Create intermediate output tensors and add to graph: Value_Info
        vi_matmul_qk = helper.make_tensor_value_info(out_name_matmul_qk, TensorProto.FLOAT,
                                                     [self.batch_size, self.num_heads, 'sequence_length',
                                                      self.total_seq_len])
        vi_add_qk_attn_mask = helper.make_tensor_value_info(out_name_add_qk_attn_mask, TensorProto.FLOAT,
                                                            [self.batch_size, self.num_heads, 'sequence_length',
                                                             self.total_seq_len])
        vi_div_qk = helper.make_tensor_value_info(out_name_div_qk, TensorProto.FLOAT,
                                                  [self.batch_size, self.num_heads, 'sequence_length',
                                                   self.total_seq_len])
        vi_softmax_qk = helper.make_tensor_value_info(out_name_softmax_qk, TensorProto.FLOAT,
                                                      [self.batch_size, self.num_heads, 'sequence_length',
                                                       self.total_seq_len])
        vi_matmul_attnv = helper.make_tensor_value_info(out_name_matmul_attnv, TensorProto.FLOAT,
                                                        [self.batch_size, self.num_heads, 'sequence_length',
                                                         self.head_dim])

        # Add new value-Infos to graph
        self.model.graph.value_info.extend([vi_matmul_qk, vi_add_qk_attn_mask, vi_div_qk,
                                            vi_softmax_qk, vi_matmul_attnv])

        # Add created nodes to graph
        self.model.graph.node.extend([node_matmul_qk, node_add_qk_attn_mask, node_constant_div_qk,
                                      node_div_qk, node_softmax_qk, node_matmul_attnv])

        return out_name_matmul_attnv

    def decompose(self):
        """
            Utility function that decomposes ort-genai generated GroupQueryAttention(GQA) op into constituent ops.
        """

        def update_o_proj_input(input_node_name: str, input_tensor_name: str, input_suffix_to_replace: str,
                                o_proj_node_suffix: str):
            o_proj_node_name = input_node_name.replace(input_suffix_to_replace, o_proj_node_suffix)

            for curr_node in self.model.graph.node:
                if curr_node.name == o_proj_node_name:
                    curr_node.input[0] = input_tensor_name

        if self.generate_conv_model:
            shape_q = np.array([self.batch_size, self.num_heads, self.head_dim, -1], dtype=np.int64)
            shape_k = np.array([self.batch_size, self.num_kv_heads, self.head_dim, -1], dtype=np.int64)
            shape_v = np.array([self.batch_size, self.num_kv_heads, self.head_dim, -1], dtype=np.int64)
            perm_order = [0, 1, 3, 2]
        else:
            shape_q = np.array([self.batch_size, -1, self.num_heads, self.head_dim], dtype=np.int64)
            shape_k = np.array([self.batch_size, -1, self.num_kv_heads, self.head_dim], dtype=np.int64)
            shape_v = np.array([self.batch_size, -1, self.num_kv_heads, self.head_dim], dtype=np.int64)
            perm_order = [0, 2, 1, 3]

        shape_attnv = np.array([-1, self.cur_seq_len, self.hidden_size], dtype=np.int64)
        expand_k_val = np.ones([self.batch_size, self.num_kv_heads, self.n_rep, self.head_dim, self.total_seq_len], dtype=np.float32)
        expand_v_val = np.ones([self.batch_size, self.num_kv_heads, self.n_rep, self.total_seq_len, self.head_dim], dtype=np.float32)
        name_init_expand_k = "expand_k_coeff"
        name_init_expand_v = "expand_v_coeff"
        init_expand_k = helper.make_tensor(name_init_expand_k,
                                           data_type=TensorProto.FLOAT,
                                           dims=expand_k_val.shape,
                                           vals=expand_k_val.flatten())
        init_expand_v = helper.make_tensor(name_init_expand_v,
                                           data_type=TensorProto.FLOAT,
                                           dims=expand_v_val.shape,
                                           vals=expand_v_val.flatten())
        self.model.graph.initializer.extend([init_expand_k, init_expand_v])
        # Iterate over nodes
        for node in self.model.graph.node:
            # check for matmul that has q, k, v weights packed into one tensor
            if node.op_type == 'GroupQueryAttention':
                # Get input output weight info
                name_gqa = node.name
                input_name_q = node.input[0]
                input_name_k = node.input[1]
                input_name_v = node.input[2]

                input_name_past_key = node.input[3]
                input_name_past_value = node.input[4]
                output_name_gqa = node.output[0]
                output_name_present_key = node.output[1]
                output_name_present_value = node.output[2]

                # Set Node Names to be added to graph
                split_name_gqa = name_gqa.split(ONNX_TENSOR_NAME_STRINGS["llama_GroupQueryAttention"])
                node_name_shape_q = split_name_gqa[0] + 'shape_q'
                node_name_shape_k = split_name_gqa[0] + 'shape_k'
                node_name_shape_v = split_name_gqa[0] + 'shape_v'
                node_name_shape_attnv = split_name_gqa[0] + 'shape_attnv'
                node_name_reshape_q = split_name_gqa[0] + 'q_reshape'
                node_name_reshape_k = split_name_gqa[0] + 'k_reshape'
                node_name_reshape_v = split_name_gqa[0] + 'v_reshape'
                node_name_transpose_q = split_name_gqa[0] + 'q_transpose'
                node_name_transpose_k = split_name_gqa[0] + 'k_transpose'
                node_name_transpose_v = split_name_gqa[0] + 'v_transpose'
                node_name_transpose_rope_k = split_name_gqa[0] + 'k_rope_transpose'
                node_name_transpose_attnv = split_name_gqa[0] + 'transpose_attnv'
                node_name_reshape_attnv = split_name_gqa[0] + 'reshape_attnv'

                # Set Output names for nodes
                out_name_shape_q = node_name_shape_q + '/output_0'
                out_name_shape_k = node_name_shape_k + '/output_0'
                out_name_shape_v = node_name_shape_v + '/output_0'
                out_name_shape_attnv = node_name_shape_attnv + '/output_0'
                out_name_reshape_q = node_name_reshape_q + '/output_0'
                out_name_reshape_k = node_name_reshape_k + '/output_0'
                out_name_reshape_v = node_name_reshape_v + '/output_0'
                out_name_transpose_q = node_name_transpose_q + '/output_0'
                out_name_transpose_k = node_name_transpose_k + '/output_0'
                out_name_transpose_v = node_name_transpose_v + '/output_0'
                out_name_transpose_attnv = node_name_transpose_attnv + '/output_0'
                out_name_reshape_attnv = node_name_reshape_attnv + '/output_0'

                # Create Decomposed GQA Nodes
                # Reshape Nodes
                node_constant_shape_q = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_q],
                                                         value=helper.make_tensor(
                                                             name=node_name_shape_q,
                                                             data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                             dims=shape_q.shape,
                                                             vals=shape_q.flatten()))
                node_constant_shape_k = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_k],
                                                         value=helper.make_tensor(
                                                             name=node_name_shape_k,
                                                             data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                             dims=shape_k.shape,
                                                             vals=shape_k.flatten()))
                node_constant_shape_v = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_v],
                                                         value=helper.make_tensor(
                                                             name=node_name_shape_v,
                                                             data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                             dims=shape_v.shape,
                                                             vals=shape_v.flatten()))
                node_reshape_q = helper.make_node('Reshape', name=node_name_reshape_q,
                                                  inputs=[input_name_q, out_name_shape_q],
                                                  outputs=[out_name_reshape_q])
                node_reshape_k = helper.make_node('Reshape', name=node_name_reshape_k,
                                                  inputs=[input_name_k, out_name_shape_k],
                                                  outputs=[out_name_reshape_k])
                node_reshape_v = helper.make_node('Reshape', name=node_name_reshape_v,
                                                  inputs=[input_name_v, out_name_shape_v],
                                                  outputs=[out_name_reshape_v])

                # Transpose Nodes
                node_transpose_q = helper.make_node('Transpose', name=node_name_transpose_q,
                                                    inputs=[out_name_reshape_q],
                                                    outputs=[out_name_transpose_q], perm=perm_order)
                node_transpose_k = helper.make_node('Transpose', name=node_name_transpose_k,
                                                    inputs=[out_name_reshape_k],
                                                    outputs=[out_name_transpose_k], perm=perm_order)
                node_transpose_v = helper.make_node('Transpose', name=node_name_transpose_v,
                                                    inputs=[out_name_reshape_v],
                                                    outputs=[output_name_present_value], perm=perm_order)

                # RoPE Nodes for Q and K
                out_name_rope_q = self.__create_rope("q", split_name_gqa[0], out_name_transpose_q,
                                                     self.pos_ids_sin_name, self.pos_ids_cos_name)
                out_name_rope_k = self.__create_rope("k", split_name_gqa[0], out_name_transpose_k,
                                                     self.pos_ids_sin_name, self.pos_ids_cos_name)

                node_transpose_rope_k = helper.make_node('Transpose', name=node_name_transpose_rope_k,
                                                         inputs=[out_name_rope_k],
                                                         outputs=[output_name_present_key], perm=[0, 1, 3, 2])

                # GQA Repetition Nodes for K and V
                out_name_reshape_current_k = self.__add_repetition_nodes("k", split_name_gqa[0],
                                                                         input_name_past_key, output_name_present_key, name_init_expand_k)
                out_name_reshape_current_v = self.__add_repetition_nodes("v", split_name_gqa[0],
                                                                         input_name_past_value, output_name_present_value, name_init_expand_v)

                # Add Scaled Dot Product Attention Nodes (softmax(QKT/sqrt(head_dim))*V)
                out_name_matmul_attnv = self.__scaled_dot_product_attention(split_name_gqa[0], out_name_rope_q,
                                                                            out_name_reshape_current_k,
                                                                            out_name_reshape_current_v)
                node_transpose_attnv = helper.make_node('Transpose', name=node_name_transpose_attnv,
                                                        inputs=[out_name_matmul_attnv],
                                                        outputs=[out_name_transpose_attnv], perm=[0, 2, 1, 3])
                if self.generate_conv_model:
                    vi_reshape_q = helper.make_tensor_value_info(out_name_reshape_q, TensorProto.FLOAT,
                                                                 [self.batch_size, self.num_heads, self.head_dim,
                                                                  'sequence_length'])
                    vi_reshape_k = helper.make_tensor_value_info(out_name_reshape_k, TensorProto.FLOAT,
                                                                 [self.batch_size, self.num_kv_heads, self.head_dim,
                                                                  'sequence_length'])
                    vi_reshape_v = helper.make_tensor_value_info(out_name_reshape_v, TensorProto.FLOAT,
                                                                 [self.batch_size, self.num_kv_heads, self.head_dim,
                                                                  'sequence_length'])
                    node_name_input_to_o_proj = node_name_transpose_attnv
                    tensor_name_input_to_o_proj = out_name_transpose_attnv
                    suffix_to_replace = "transpose_attnv"
                    o_proj_node_name_suffix = "o_proj/PreConvReshape"
                else:
                    # If not generating a Conv model (creating a model with MatMul),
                    # then we need to reshape the output of the Scaled Dot Product Attention.
                    # Add Reshape Node at GQA end
                    node_constant_shape_attnv = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_attnv],
                                                                 value=helper.make_tensor(
                                                                     name=node_name_shape_attnv,
                                                                     data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                                     dims=shape_attnv.shape,
                                                                     vals=shape_attnv.flatten()))
                    node_reshape_attnv = helper.make_node('Reshape', name=node_name_reshape_attnv,
                                                          inputs=[out_name_transpose_attnv, out_name_shape_attnv],
                                                          outputs=[out_name_reshape_attnv])
                    vi_reshape_q = helper.make_tensor_value_info(out_name_reshape_q, TensorProto.FLOAT,
                                                                 [self.batch_size, 'sequence_length', self.num_heads,
                                                                  self.head_dim])
                    vi_reshape_k = helper.make_tensor_value_info(out_name_reshape_k, TensorProto.FLOAT,
                                                                 [self.batch_size, 'sequence_length', self.num_kv_heads,
                                                                  self.head_dim])
                    vi_reshape_v = helper.make_tensor_value_info(out_name_reshape_v, TensorProto.FLOAT,
                                                                 [self.batch_size, 'sequence_length', self.num_kv_heads,
                                                                  self.head_dim])
                    vi_reshape_attnv = helper.make_tensor_value_info(out_name_reshape_attnv, TensorProto.FLOAT,
                                                                     [self.batch_size, 'sequence_length',
                                                                      self.hidden_size])
                    node_name_input_to_o_proj = node_name_reshape_attnv
                    tensor_name_input_to_o_proj = out_name_reshape_attnv
                    suffix_to_replace = "reshape_attnv"
                    o_proj_node_name_suffix = "o_proj/MatMul"
                    self.model.graph.value_info.extend([vi_reshape_attnv])
                    self.model.graph.node.extend([node_constant_shape_attnv, node_reshape_attnv])

                # Create intermediate output tensors and add to graph: Value_Info
                vi_transpose_q = helper.make_tensor_value_info(out_name_transpose_q, TensorProto.FLOAT,
                                                               [self.batch_size, self.num_heads, 'sequence_length',
                                                                self.head_dim])
                vi_transpose_k = helper.make_tensor_value_info(out_name_transpose_k, TensorProto.FLOAT,
                                                               [self.batch_size, self.num_kv_heads, 'sequence_length',
                                                                self.head_dim])
                vi_transpose_v = helper.make_tensor_value_info(out_name_transpose_v, TensorProto.FLOAT,
                                                               [self.batch_size, self.num_kv_heads, 'sequence_length',
                                                                self.head_dim])
                vi_transpose_attnv = helper.make_tensor_value_info(out_name_transpose_attnv, TensorProto.FLOAT,
                                                                   [self.batch_size, 'sequence_length', self.num_heads,
                                                                    self.head_dim])

                self.model.graph.value_info.extend([vi_reshape_q, vi_reshape_k, vi_reshape_v,
                                                    vi_transpose_q, vi_transpose_k, vi_transpose_v,
                                                    vi_transpose_attnv])

                # Add created nodes to graph
                self.model.graph.node.extend([node_constant_shape_q, node_constant_shape_k, node_constant_shape_v,
                                              node_reshape_q, node_reshape_k, node_reshape_v,
                                              node_transpose_q, node_transpose_k, node_transpose_v,
                                              node_transpose_rope_k, node_transpose_attnv])

                # Remove GQA Node and Value Info
                gqa_vi = [vi for vi in self.model.graph.value_info if vi.name == output_name_gqa][0]
                self.model.graph.value_info.remove(gqa_vi)
                self.model.graph.node.remove(node)

                update_o_proj_input(node_name_input_to_o_proj, tensor_name_input_to_o_proj, suffix_to_replace,
                                    o_proj_node_name_suffix)


def find_add_node_following_matmul(model, matmul_node):
    matmul_output_name = matmul_node.output[0]
    add_node = None
    for node in model.graph.node:
        if node.op_type == "Add" and node.input[0] == matmul_output_name:
            add_node = node
    return add_node


class LinearToConv:
    """
    LinearToConv Utility Class to convert Linear/MatMul Nodes in graph with Convolution Nodes.
    """
    def __init__(self, model: ModelProto, gguf_onnx_config: GGUFONNXConfig):
        # Update the configuration to generate a convolution model
        gguf_onnx_config.update(generate_conv_model=True)
        # Initialize instance variables
        self.cur_seq_len = gguf_onnx_config.arn_seq_len
        self.batch_size = gguf_onnx_config.batch_size
        self.model = model
        self.hidden_size = gguf_onnx_config.hidden_size
        self.vocab_size = gguf_onnx_config.vocab_size
        self.head_dim = gguf_onnx_config.head_dim
        self.num_heads = gguf_onnx_config.num_heads
        self.num_kv_heads = gguf_onnx_config.num_kv_heads
        if self.vocab_size is None:
            for vi in model.graph.output:
                if vi.name == "logits":
                    dims = vi.type.tensor_type.shape.dim
                    self.vocab_size = dims[-1].dim_value
        # Define the projection names for attention and MLP layers
        self.attn_proj = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.mlp_proj = ["up_proj", "down_proj", "gate_proj"]

    def __update_weight_initializer(self, weight_name_proj):
        """
        Update the weight initializer for a convolution node.

        Args:
        weight_name_proj (str): The name of the weight initializer to update.

        Returns:
        str: The updated weight name.
        tuple: The shape of the updated weight.
        """
        # Find the weight initializer in the model graph
        weight_init_proj = next((initializer for initializer in self.model.graph.initializer if initializer.name == weight_name_proj), None)
        if weight_init_proj is None:
            # Raise an error if the weight initializer is not found
            raise ValueError(f"Weight initializer not found for {weight_name_proj}")

        # Get the weight tensor from the initializer
        weight_tensor_proj = numpy_helper.to_array(weight_init_proj)  # I x O

        # Transpose and reshape the weight tensor for convolution
        weight_tensor_proj_conv = np.transpose(weight_tensor_proj)
        exp_shape = (weight_tensor_proj_conv.shape[0], weight_tensor_proj_conv.shape[1], 1, 1)  # O x I x H x W
        weight_tensor_proj_conv = np.reshape(weight_tensor_proj_conv, exp_shape)

        # Update the weight name and remove the old initializer
        split_weight_name_proj = weight_name_proj.split("MatMul")
        weight_name_proj_conv = split_weight_name_proj[0] + "Conv" + split_weight_name_proj[1]
        self.model.graph.initializer.remove(weight_init_proj)

        # Create a new initializer for the updated weight
        init_proj_conv = numpy_helper.from_array(weight_tensor_proj_conv, weight_name_proj_conv)
        # add initializer to graph
        self.model.graph.initializer.extend([init_proj_conv])

        return weight_name_proj_conv, exp_shape

    def __update_bias_initializer(self, bias_name_proj):
        """
        Update the bias initializer for a convolution node.

        Args:
        bias_name_proj (str): The name of the bias initializer to update.

        Returns:
        str: The updated bias name.
        """
        # Find the bias initializer in the model graph
        bias_init_proj = next((initializer for initializer in self.model.graph.initializer if initializer.name == bias_name_proj), None)
        if bias_init_proj is None:
            # Raise an error if the bias initializer is not found
            raise ValueError(f"Bias initializer not found for {bias_name_proj}")

        # Update the bias name
        split_bias_name_proj = bias_name_proj.split("Add")
        bias_name_proj_conv = split_bias_name_proj[0] + "Conv" + split_bias_name_proj[1]
        bias_init_proj.name = bias_name_proj_conv
        return bias_name_proj_conv

    def __conv_preprocess(self, input_name_proj, split_name_proj):
        """
        Preprocess the input for a convolution node.

        Args:
        input_name_proj (str): The name of the input tensor.
        split_name_proj (list): The split name of the projection.

        Returns:
        str: The name of the preprocessed input tensor.
        """
        # Determine the output size based on the projection type
        if "o_proj" in split_name_proj[0]:
            out_size = self.head_dim*self.num_heads
        else:
            out_size = self.hidden_size
        # Define the shape of the input tensor
        shape_proj = np.array([self.batch_size, self.cur_seq_len, 1, out_size], dtype=np.int64)

        node_name_shape_proj = split_name_proj[0] + "PreConvConstant"
        node_name_proj_reshape = split_name_proj[0] + "PreConvReshape"
        node_name_proj_transpose = split_name_proj[0] + "PreConvTranspose"

        out_name_shape_proj = node_name_shape_proj + "/output_0"
        out_name_proj_reshape = node_name_proj_reshape + "/output_0"
        out_name_proj_transpose = node_name_proj_transpose + "/output_0"

        node_constant_shape_proj = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_proj],
                                                    value=helper.make_tensor(
                                                        name=node_name_shape_proj,
                                                        data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                        dims=shape_proj.shape,
                                                        vals=shape_proj.flatten()))
        node_proj_reshape = helper.make_node('Reshape', name=node_name_proj_reshape,
                                             inputs=[input_name_proj, out_name_shape_proj],
                                             outputs=[out_name_proj_reshape])
        node_proj_transpose = helper.make_node("Transpose", name=node_name_proj_transpose,
                                               inputs=[out_name_proj_reshape],
                                               outputs=[out_name_proj_transpose], perm=[0, 3, 2, 1])

        # Add value information for the new preprocessing nodes tensors
        vi_proj_reshape = helper.make_tensor_value_info(out_name_proj_reshape, TensorProto.FLOAT,
                                                        [self.batch_size, self.cur_seq_len, 1, out_size])
        vi_proj_transpose = helper.make_tensor_value_info(out_name_proj_transpose, TensorProto.FLOAT,
                                                          [self.batch_size, out_size, 1, self.cur_seq_len])
        self.model.graph.value_info.extend([vi_proj_reshape, vi_proj_transpose])

        # add nodes to graph
        self.model.graph.node.extend([node_constant_shape_proj, node_proj_reshape, node_proj_transpose])
        return out_name_proj_transpose

    def __conv_postprocess(self, out_name_proj_conv, output_name_proj, split_name_proj):
        """
        Postprocess the output of a convolution node.

        Args:
        out_name_proj_conv (str): The name of the output tensor.
        output_name_proj (str): The name of the final output tensor.
        split_name_proj (list): The split name of the projection.
        """
        # Determine the output size based on the projection type
        if any([val for val in split_name_proj if "lm_head" in val]):
            out_size = self.vocab_size
        else:
            out_size = self.hidden_size

        # Define the shape of the output tensor
        shape_proj = np.array([self.batch_size, self.cur_seq_len, out_size], dtype=np.int64)

        node_name_proj_transpose = split_name_proj[0] + "PostConvTranspose"
        node_name_shape_proj = split_name_proj[0] + "PostConvConstant"
        node_name_proj_reshape = split_name_proj[0] + "PostConvReshape"

        out_name_proj_transpose = node_name_proj_transpose + "/output_0"
        out_name_shape_proj = node_name_shape_proj + "/output_0"

        node_proj_transpose = helper.make_node("Transpose", name=node_name_proj_transpose, inputs=[out_name_proj_conv],
                                               outputs=[out_name_proj_transpose], perm=[0, 3, 2, 1])
        node_constant_shape_proj = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_proj],
                                                    value=helper.make_tensor(
                                                        name=node_name_shape_proj,
                                                        data_type=helper.np_dtype_to_tensor_dtype(np.dtype('int64')),
                                                        dims=shape_proj.shape,
                                                        vals=shape_proj.flatten()))
        node_proj_reshape = helper.make_node('Reshape', name=node_name_proj_reshape,
                                             inputs=[out_name_proj_transpose, out_name_shape_proj],
                                             outputs=[output_name_proj])

        # Add value information for the new postprocessing nodes tensors
        vi_proj_transpose = helper.make_tensor_value_info(out_name_proj_transpose, TensorProto.FLOAT,
                                                          [self.batch_size, self.cur_seq_len, 1, out_size])
        self.model.graph.value_info.extend([vi_proj_transpose])
        # add nodes to graph
        self.model.graph.node.extend([node_proj_transpose, node_constant_shape_proj, node_proj_reshape])

    def __linear_to_conv_node_helper(self, node, add_node):
        """
        Helper function to convert a linear node to a convolution node.

        Args:
        node (NodeProto): The linear node to convert.
        add_node (NodeProto): The add node associated with the linear node.
        """
        def update_proj_input(node_name: str, output_tensor_name: str, cur_substr, exp_substr):
            """
            Update the input of a projection node.

            Args:
            node_name (str): The name of the projection node.
            output_tensor_name (str): The name of the output tensor.
            cur_substr (str): The current substring to replace.
            exp_substr (str): The expected substring to replace with.
            """
            proj_node_name = node_name.replace(cur_substr, exp_substr)
            for curr_node in self.model.graph.node:
                if curr_node.name == proj_node_name:
                    curr_node.input[0] = output_tensor_name

        # get node, input, output and weight names
        name_proj = node.name
        input_name_proj = node.input[0]
        output_name_proj = node.output[0] if add_node is None else add_node.output[0]
        weight_name_proj = node.input[1]

        # Update the weight initializer
        weight_name_proj_conv, weight_shape = self.__update_weight_initializer(weight_name_proj)

        # set names for proj Conv node
        split_name_proj = name_proj.split("MatMul")
        node_name_proj_conv = split_name_proj[0] + "Conv"

        # Update the bias initializer if an add node is provided
        bias_name = None
        if add_node is not None:
            bias_name = self.__update_bias_initializer(add_node.input[1])
        bias_name_proj_conv = [] if add_node is None else [bias_name]

        # Initialize the input and output names for the convolution node
        in_name_proj_conv = ""
        out_name_proj_conv = ""
        # create Nodes

        # New Sub-Graph for gate_proj: Reshape-> Transpose -> Conv (Gate_Proj)
        if "gate_proj" in node.name:
            in_name_proj_conv = self.__conv_preprocess(input_name_proj, split_name_proj)
            out_name_proj_conv = output_name_proj
            # Set input of up_proj to output of Transpose added in Gate_Proj path
            update_proj_input(node_name_proj_conv, in_name_proj_conv, "gate_proj/Conv", "up_proj/MatMul")

        # New Sub-Graph for up_proj : Conv (Up_Proj)
        elif "up_proj" in node.name:
            in_name_proj_conv = input_name_proj
            out_name_proj_conv = output_name_proj

        # New Sub-Graph for down_proj : Conv (Down_Proj) -> Transpose -> Reshape
        elif "down_proj" in node.name:
            in_name_proj_conv = input_name_proj
            out_name_proj_conv = node_name_proj_conv + "/output_0"
            self.__conv_postprocess(out_name_proj_conv, output_name_proj, split_name_proj)

        # New Sub-Graph for q_proj :  Reshape -> Transpose -> Conv (q_Proj) -> Transpose
        elif "q_proj" in node.name:
            in_name_proj_conv = self.__conv_preprocess(input_name_proj, split_name_proj)
            out_name_proj_conv = output_name_proj
            # out_name_proj_conv = node_name_proj_conv + "/output_0"
            # Set input of k_proj,v_proj to output of Transpose added in q_Proj path
            update_proj_input(node_name_proj_conv, in_name_proj_conv, "q_proj/Conv", "k_proj/MatMul")
            update_proj_input(node_name_proj_conv, in_name_proj_conv, "q_proj/Conv", "v_proj/MatMul")

        # New Sub-Graph for k_proj : Conv (k_Proj) -> Transpose
        elif "k_proj" in node.name:
            in_name_proj_conv = input_name_proj
            out_name_proj_conv = output_name_proj

        # New Sub-Graph for v_proj : Conv (v_Proj) -> Transpose
        elif "v_proj" in node.name:
            in_name_proj_conv = input_name_proj
            out_name_proj_conv = output_name_proj

        # New Sub-Graph for o_proj :  Reshape -> Transpose -> Conv (o_Proj) -> Transpose -> Reshape
        elif "o_proj" in node.name:
            in_name_proj_conv = self.__conv_preprocess(input_name_proj, split_name_proj)
            out_name_proj_conv = node_name_proj_conv + "/output_0"
            self.__conv_postprocess(out_name_proj_conv, output_name_proj, split_name_proj)

        # New Sub-Graph for lm_head :  Reshape -> Transpose -> Conv (lm_head) -> Transpose -> Reshape
        elif "lm_head" in node.name:
            in_name_proj_conv = self.__conv_preprocess(input_name_proj, split_name_proj)
            out_name_proj_conv = node_name_proj_conv + "/output_0"
            self.__conv_postprocess(out_name_proj_conv, output_name_proj, split_name_proj)

        # Create a convolution node
        node_proj_conv = helper.make_node("Conv", name=node_name_proj_conv,
                                          inputs=[in_name_proj_conv, weight_name_proj_conv] + bias_name_proj_conv,
                                          outputs=[out_name_proj_conv])

        # update value info
        # Remove the ValueInfo for gate/up as they are 3d but Conv output is 4d
        # (o_proj/down_proj/lm_head output vi is retained)
        proj_nodes = ["gate_proj", "up_proj"]
        if any([proj_str for proj_str in proj_nodes if proj_str in node.name]):
            vi_existing_proj = [vi for vi in self.model.graph.value_info if vi.name == output_name_proj][0]
            self.model.graph.value_info.remove(vi_existing_proj)

        # Add value information for the convolution output tensor
        vi_proj_conv = helper.make_tensor_value_info(out_name_proj_conv, TensorProto.FLOAT,
                                                     [self.batch_size, weight_shape[0], 1, self.cur_seq_len])
        self.model.graph.value_info.extend([vi_proj_conv])

        # add nodes to graph
        self.model.graph.node.extend([node_proj_conv])
        # Remove Current MatMul node from Graph
        self.model.graph.node.remove(node)
        # Remove Add node from Graph
        if add_node is not None:
            self.model.graph.node.remove(add_node)

    def __perform_linear_to_conv_nodes(self, nodes_list, find_add_node: bool):
        """
            Utility function to convert linear(FC) layers to Convolution Layers.
            Args:
            nodes_list (list): The list of linear nodes to convert.
            find_add_node (bool): Whether to find an add node following the linear node.

            q_proj, k_proj, v_proj and o_proj are converted to conv ops for running optimally on Qualcomm HTP target.
            gate_proj, up_proj and down_proj are converted to conv ops for running optimally on Qualcomm HTP target.
            lm_head linear is converted to conv ops for running optimally on Qualcomm HTP target.
        """
        add_node = None
        for node in nodes_list:
            if find_add_node:
                # Find an add node following the linear node
                add_node = find_add_node_following_matmul(self.model, node)
            # Convert the linear node to a convolution node
            self.__linear_to_conv_node_helper(node, add_node)

    def convert(self):
        """
        Convert linear nodes to convolution nodes.
        """
        attn_nodes_list = []
        mlp_nodes_list = []
        lm_head_node_list = []
        # Iterate over nodes to find MatMul nodes and cache in attn/mlp list
        for node in self.model.graph.node:
            if node.op_type == "MatMul" and any([val for val in self.attn_proj if val in node.name]):
                attn_nodes_list.append(node)
            if node.op_type == "MatMul" and any([val for val in self.mlp_proj if val in node.name]):
                mlp_nodes_list.append(node)
            if node.op_type == "MatMul" and "lm_head" in node.name:
                lm_head_node_list.append(node)
        self.__perform_linear_to_conv_nodes(attn_nodes_list, True)
        self.__perform_linear_to_conv_nodes(mlp_nodes_list, False)
        self.__perform_linear_to_conv_nodes(lm_head_node_list, False)

        vi_mlp_sigmoid_mul = [vi for vi in self.model.graph.value_info if "Sigmoid" in vi.name]
        vi_mlp_sigmoid_mul += [vi for vi in self.model.graph.value_info if "mlp/act_fn/Mul" in vi.name]
        vi_mlp_sigmoid_mul += [vi for vi in self.model.graph.value_info if "mlp/Mul" in vi.name]
        for vi in vi_mlp_sigmoid_mul:
            self.model.graph.value_info.remove(vi)


def unpack_qkv(model: ModelProto, gguf_onnx_config: GGUFONNXConfig):
    """
        Utility function to subdivide ort-genai generated combined QKV FullyConnected(FC).
        Combined QKV op is split into 3 FC Ops for Q,K and V respectively.
    """

    def update_gqa_inputs(model: ModelProto, node_name: str, output_tensor_name: str, node_type: str):
        gqa_node_name = node_name.replace(f"q_proj/{node_type}", "GroupQueryAttention")

        for node in model.graph.node:
            if node.name == gqa_node_name:
                node.input[0] = output_tensor_name
                node.input[1] = output_tensor_name.replace("q_proj", "k_proj")
                node.input[2] = output_tensor_name.replace("q_proj", "v_proj")

    hidden_size = gguf_onnx_config.hidden_size
    num_heads = gguf_onnx_config.num_heads
    num_kv_heads = gguf_onnx_config.num_kv_heads
    head_dim = gguf_onnx_config.head_dim
    n_q = num_heads * head_dim
    n_k = num_kv_heads * head_dim
    n_v = num_kv_heads * head_dim

    # Iterate over nodes
    for node in model.graph.node:

        # check for matmul that has q, k, v weights packed into one tensor
        if node.op_type == "MatMul" and "qkv_proj" in node.name:

            # get input, output and weight names
            name_qkv = node.name

            matmul_layer_idx = re.findall(r"layers\.\d+", name_qkv)[0].split(".")[-1]
            input_name_qkv = node.input[0]
            output_name_qkv = node.output[0]
            weight_name_qkv = node.input[1]

            # get weight initializer
            weight_init_qkv = \
            [initializer for initializer in model.graph.initializer if weight_name_qkv == initializer.name][0]

            # get weight tensor
            weight_tensor_qkv = numpy_helper.to_array(weight_init_qkv)

            # split packed tensor into wq, wk, wv
            weight_tensor_q = np.copy(weight_tensor_qkv[:, : n_q])
            weight_tensor_k = np.copy(weight_tensor_qkv[:, n_q: n_q + n_k])
            weight_tensor_v = np.copy(weight_tensor_qkv[:, n_q + n_k: n_q + n_k + n_v])

            # remove packed tensor
            model.graph.initializer.remove(weight_init_qkv)

            # set wq, wk, wv tensor names
            split_w_qkv = weight_name_qkv.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
            weight_name_q = split_w_qkv[0] + "q_proj" + split_w_qkv[1]
            weight_name_k = split_w_qkv[0] + "k_proj" + split_w_qkv[1]
            weight_name_v = split_w_qkv[0] + "v_proj" + split_w_qkv[1]

            # set names for q, k, v Matmul nodes
            split_name_qkv = name_qkv.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
            node_name_q = split_name_qkv[0] + "q_proj" + split_name_qkv[1]
            node_name_k = split_name_qkv[0] + "k_proj" + split_name_qkv[1]
            node_name_v = split_name_qkv[0] + "v_proj" + split_name_qkv[1]

            # create initializers from split tensors
            init_q = numpy_helper.from_array(weight_tensor_q, weight_name_q)
            init_k = numpy_helper.from_array(weight_tensor_k, weight_name_k)
            init_v = numpy_helper.from_array(weight_tensor_v, weight_name_v)

            # add split tensors to initializers
            model.graph.initializer.extend([init_q, init_k, init_v])

            # set output names for split nodes
            out_name_q = node_name_q + "/output_0"
            out_name_k = node_name_k + "/output_0"
            out_name_v = node_name_v + "/output_0"

            # create split nodes
            node_matmul_q = helper.make_node("MatMul", name=node_name_q, inputs=[input_name_qkv, weight_name_q],
                                             outputs=[out_name_q])

            node_matmul_k = helper.make_node("MatMul", name=node_name_k, inputs=[input_name_qkv, weight_name_k],
                                             outputs=[out_name_k])

            node_matmul_v = helper.make_node("MatMul", name=node_name_v, inputs=[input_name_qkv, weight_name_v],
                                             outputs=[out_name_v])

            # create output tensors and add to graph
            q_vi = helper.make_tensor_value_info(out_name_q, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_q.shape[-1]])
            k_vi = helper.make_tensor_value_info(out_name_k, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_k.shape[-1]])
            v_vi = helper.make_tensor_value_info(out_name_v, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_v.shape[-1]])

            qkv_vi = [vi for vi in model.graph.value_info if vi.name == output_name_qkv][0]

            model.graph.value_info.extend([q_vi, k_vi, v_vi])
            model.graph.value_info.remove(qkv_vi)

            # add split nodes to graph
            model.graph.node.extend([node_matmul_q, node_matmul_k, node_matmul_v])
            model.graph.node.remove(node)

            gqa_in_node_name = node_name_q
            out_tensor_name = out_name_q
            node_type_str = "MatMul"

            for next_node in model.graph.node:
                if next_node.op_type == "Add" and "qkv_proj" in next_node.name and \
                        re.findall(r"layers\.\d+", next_node.name)[0].split(".")[-1] == matmul_layer_idx:
                    # Get names of input, output and weight
                    name_qkv_add = next_node.name
                    output_qkv_add = next_node.output[0]
                    weight_qkv_add = next_node.input[1]

                    # get weight initializer
                    weight_init_qkv_add = \
                    [initializer for initializer in model.graph.initializer if weight_qkv_add == initializer.name][0]

                    # get weight tensor
                    weight_tensor_qkv_add = numpy_helper.to_array(weight_init_qkv_add)

                    # split packed tensor into wq, wk, wv
                    weight_tensor_q_add = np.copy(weight_tensor_qkv_add[: n_q])
                    weight_tensor_k_add = np.copy(weight_tensor_qkv_add[n_q: n_q + n_k])
                    weight_tensor_v_add = np.copy(weight_tensor_qkv_add[n_q + n_k: n_q + n_k + n_v])

                    # remove packed tensor
                    model.graph.initializer.remove(weight_init_qkv_add)

                    # set wq, wk, wv tensor names
                    split_w_qkv_add = weight_qkv_add.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
                    weight_name_q_add = split_w_qkv_add[0] + "q_proj" + split_w_qkv_add[1]
                    weight_name_k_add = split_w_qkv_add[0] + "k_proj" + split_w_qkv_add[1]
                    weight_name_v_add = split_w_qkv_add[0] + "v_proj" + split_w_qkv_add[1]

                    # set names for q, k, v Add nodes
                    split_name_qkv_add = name_qkv_add.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
                    node_name_q_add = split_name_qkv_add[0] + "q_proj" + split_name_qkv_add[1]
                    node_name_k_add = split_name_qkv_add[0] + "k_proj" + split_name_qkv_add[1]
                    node_name_v_add = split_name_qkv_add[0] + "v_proj" + split_name_qkv_add[1]

                    # create initializers from split tensors
                    init_q_add = numpy_helper.from_array(weight_tensor_q_add, weight_name_q_add)
                    init_k_add = numpy_helper.from_array(weight_tensor_k_add, weight_name_k_add)
                    init_v_add = numpy_helper.from_array(weight_tensor_v_add, weight_name_v_add)

                    # add split tensors to initializers
                    model.graph.initializer.extend([init_q_add, init_k_add, init_v_add])

                    # set output names for split nodes
                    out_name_q_add = node_name_q_add + "/output_0"
                    out_name_k_add = node_name_k_add + "/output_0"
                    out_name_v_add = node_name_v_add + "/output_0"

                    # create split nodes
                    node_add_q = helper.make_node("Add", name=node_name_q_add, inputs=[out_name_q, weight_name_q_add],
                                                  outputs=[out_name_q_add])

                    node_add_k = helper.make_node("Add", name=node_name_k_add, inputs=[out_name_k, weight_name_k_add],
                                                  outputs=[out_name_k_add])

                    node_add_v = helper.make_node("Add", name=node_name_v_add, inputs=[out_name_v, weight_name_v_add],
                                                  outputs=[out_name_v_add])

                    # create output tensors and add to graph
                    q_vi_add = helper.make_tensor_value_info(out_name_q_add, TensorProto.FLOAT,
                                                             ["batch_size", "sequence_length",
                                                              weight_tensor_q_add.shape[-1]])
                    k_vi_add = helper.make_tensor_value_info(out_name_k_add, TensorProto.FLOAT,
                                                             ["batch_size", "sequence_length",
                                                              weight_tensor_k_add.shape[-1]])
                    v_vi_add = helper.make_tensor_value_info(out_name_v_add, TensorProto.FLOAT,
                                                             ["batch_size", "sequence_length",
                                                              weight_tensor_v_add.shape[-1]])

                    qkv_vi_add = [vi for vi in model.graph.value_info if vi.name == output_qkv_add][0]

                    model.graph.value_info.extend([q_vi_add, k_vi_add, v_vi_add])
                    model.graph.value_info.remove(qkv_vi_add)

                    # add split nodes to graph
                    model.graph.node.extend([node_add_q, node_add_k, node_add_v])
                    model.graph.node.remove(next_node)

                    gqa_in_node_name = node_name_q_add
                    out_tensor_name = out_name_q_add
                    node_type_str = "Add"

            update_gqa_inputs(model, gqa_in_node_name, out_tensor_name, node_type_str)


def update_encoding_tensor_name_mapping_dict(generate_conv_model: bool, model_type):
    global GGUF_TO_ONNX_TENSOR
    if generate_conv_model:
        op_str = "Conv"
    else:
        op_str = "MatMul"
    for k,v in GGUF_TO_ONNX_TENSOR[model_type].items():
        if "op_type" in v:
            GGUF_TO_ONNX_TENSOR[model_type][k] = v.format(op_type=op_str)


def prune_graph(model: ModelProto):
    """
    Remove unused nodes from the ONNX model.

    Args:
        model: input ONNX model.

    Returns:
        List of unused nodes removed from the model
    """

    def find_nodes_with_missing_inputs():
        initializer_names = set(init.name for init in model.graph.initializer)
        graph_input_names = set(inp.name for inp in model.graph.input)
        node_output_names = set()
        for node in model.graph.node:
            for out in node.output:
                if out:
                    node_output_names.add(out)

        available_names = initializer_names | graph_input_names | node_output_names

        nodes_with_missing_inputs = []
        for i, node in enumerate(model.graph.node):
            missing_inputs = []
            for inp in node.input:
                if inp and inp not in available_names:
                    missing_inputs.append(inp)

            if missing_inputs:
                nodes_with_missing_inputs.append((i, node, missing_inputs))

        return nodes_with_missing_inputs

    def find_unused_outputs(node_outputs):

        graph_output_names = set(out.name for out in model.graph.output)

        used_by_nodes = set()
        for node in model.graph.node:
            for inp in node.input:
                if inp in node_outputs:
                    used_by_nodes.add(inp)

        unused = set(node_outputs) - graph_output_names - used_by_nodes
        return unused

    def find_dependent_nodes(node_outputs):
        dependent_nodes = []
        output_set = set(node_outputs)

        for i, node in enumerate(model.graph.node):
            for inp in node.input:
                if inp in output_set:
                    dependent_nodes.append((i, node))
                    break

        return dependent_nodes

    nodes_with_missing_inputs = find_nodes_with_missing_inputs()

    if not nodes_with_missing_inputs:
        return

    nodes_to_remove = []

    for idx, node, missing_inputs in nodes_with_missing_inputs:
        unused_outputs = find_unused_outputs(list(node.output))

        if unused_outputs == set(node.output):
            nodes_to_remove.append((idx, node))

        else:
            dependent_nodes = find_dependent_nodes(list(node.output))

            if dependent_nodes:
                all_deps_unused = True

                for dep_idx, dep_node in dependent_nodes:
                    dep_unused = find_unused_outputs(list(dep_node.output))
                    if dep_unused != set(dep_node.output):
                        all_deps_unused = False
                        break

                if all_deps_unused:
                    nodes_to_remove.append((idx, node))
                    nodes_to_remove.extend(dependent_nodes)

    if not nodes_to_remove:
        return

    indices_to_remove = set(idx for idx, _ in nodes_to_remove)
    remaining_nodes = [node for i, node in enumerate(model.graph.node) if i not in indices_to_remove]

    model.graph.ClearField('node')
    model.graph.node.extend(remaining_nodes)

    removed_nodes_names = [node[1].name for node in nodes_to_remove]
    logger.debug(f"Removed {len(removed_nodes_names)} unused nodes: {removed_nodes_names} from the ONNX graph")


def update_onnx_graph_helper(model: ModelProto, model_config: dict, arn_value: int, batch_size: int):
    """
    Updates the ONNX graph by applying various transformations.

    Args:
        model (ModelProto): The ONNX model to be updated.
        model_config (dict): The model configuration.
        arn_value (int): The ARN value.
        batch_size (int): The batch size.
    """
    # Create a GGUF ONNX configuration instance
    gguf_onnx_config = GGUFONNXConfig(model_config)
    # Update the GGUF ONNX configuration with the provided ARN value and batch size
    gguf_onnx_config.update(arn_seq_len=arn_value, batch_size=batch_size)

    model_io_updater = ModelInputOutputUpdater(model, gguf_onnx_config)
    model_io_updater.update_input_datatypes()
    model_io_updater.update_attention_mask()
    model_io_updater.update_kv_cache_input_output_names()
    model_io_updater.update_model_inputs_rope()

    ln_obj = LayerNormalization(model, gguf_onnx_config)
    ln_obj.decompose()
    unpack_qkv(model, gguf_onnx_config)
    l2c_op = LinearToConv(model, gguf_onnx_config)
    l2c_op.convert()
    gqa_obj = GroupQueryAttention(model, gguf_onnx_config)
    gqa_obj.decompose()
    prune_graph(model)
    update_encoding_tensor_name_mapping_dict(gguf_onnx_config.generate_conv_model, gguf_onnx_config.model_type)
    update_symbolic_shape_with_value(model, gguf_onnx_config)


def update_encodings(param_encodings: dict, num_layers: int, model_type: str):
    """
        Utility Function that updates the names of tensors in encodings.json file.
        The tensor names in encodings corresponding to GGUF are updated with ONNX model tensor names.
    """
    onnx_tensor_name_map = GGUF_TO_ONNX_TENSOR[model_type]

    for param_encoding in param_encodings:
        tensor_name = param_encoding["name"]

        for onnx_tensor_key in onnx_tensor_name_map:
            if onnx_tensor_key in tensor_name:
                onnx_tensor_value = onnx_tensor_name_map[onnx_tensor_key]
                if onnx_tensor_key == "output_norm":
                    onnx_tensor_value = onnx_tensor_value.format(max_block=num_layers)
                tensor_name = tensor_name.replace(onnx_tensor_key, onnx_tensor_value)
            param_encoding["name"] = tensor_name


def permute_weights(weights, num_heads: int, num_kv_heads: int):
    if num_kv_heads is not None and num_heads != num_kv_heads:
        num_heads = num_kv_heads
    return (weights.reshape(num_heads, 2, weights.shape[0] // num_heads // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


class ModelSection(Enum):
    """Enum representing different sections of a model for size analysis."""
    EMBEDDING = "embedding"
    HIDDEN_LAYER = "block"
    LM_HEAD = "lm_head"


def get_model_section_size_info(model_path: str) -> dict:
    """
    Calculate parameter counts for different sections of a GGUF model.
    
    This function analyzes a GGUF model file and returns the number of parameters
    (elements) in three key sections: embedding layer, a single hidden layer block,
    and the language model head.
    
    Args:
        model_path: Path to the GGUF model file.
        
    Returns:
        Dictionary mapping ModelSection enum values to parameter counts:
        - EMBEDDING: Number of parameters in the token embedding layer
        - HIDDEN_LAYER: Number of parameters in the first transformer block (blk.0)
        - LM_HEAD: Number of parameters in the output layer (including final norm)
        
    Raises:
        FileNotFoundError: If the specified GGUF file does not exist.
        
    Note:
        - Only counts parameters from the first block (blk.0) for HIDDEN_LAYER
        - Handles weight tying: if output.weight is missing, assumes embedding
          weights are shared with the output layer
    """
    # Initialize counters for each model section
    elements_counts = {
        ModelSection.EMBEDDING: 0,
        ModelSection.HIDDEN_LAYER: 0,
        ModelSection.LM_HEAD: 0
    }

    # Validate file existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Provided GGUF file does not exist: {model_path}')

    # Read GGUF file metadata (does not load full tensor data)
    reader = GGUFReader(model_path)

    # Flag to track if output layer shares weights with embedding (weight tying)
    use_embedding_for_output = True

    # Iterate through all tensors and accumulate parameter counts
    for t in reader.tensors:
        # Count parameters in first transformer block only (representative of one layer)
        if t.name.startswith('blk.0'):
            elements_counts[ModelSection.HIDDEN_LAYER] += t.n_elements
        
        # Count embedding layer parameters
        if t.name == "token_embd.weight":
            elements_counts[ModelSection.EMBEDDING] += t.n_elements
        
        # Count final layer norm parameters (part of LM head)
        if t.name == "output_norm.weight":
            elements_counts[ModelSection.LM_HEAD] += t.n_elements
        
        # Count output projection parameters if present
        if t.name == "output.weight":
            use_embedding_for_output = False
            elements_counts[ModelSection.LM_HEAD] += t.n_elements

    # Handle weight tying: if no separate output.weight, embedding is reused
    if use_embedding_for_output:
        elements_counts[ModelSection.LM_HEAD] += elements_counts[ModelSection.EMBEDDING]

    return elements_counts
