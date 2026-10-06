# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import gc
import sys
import json
import copy
import gguf
import logging
import traceback
import numpy as np

from tqdm import tqdm
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.integrations.ggml import GGML_TYPES
from . import permute_weights, MODEL_TYPE_TO_ARCH, GGUF_TENSOR_NAME_STRINGS, GGUF_TYPE_BLOCK_SIZE

logger = logging.getLogger(__name__)


class GGUFParser:
    """
    Class to parse GGUF files and export de-quantized weights and encodings.
    """
    def __init__(self, input_model: str, filename_prefix: str, output_dir: str = None):
        """
        Constructor
        :param input_model: Path to GGUF file.
        :param filename_prefix: Prefix for the exported encodings file.
        :param output_dir: Path to the output directory for saving the processed files.
        """
        self.arch = None
        self.input_model = os.path.abspath(input_model)
        self.output_dir = output_dir if output_dir else os.path.dirname(input_model)
        self.filename_prefix = filename_prefix
        self.param_info = dict()
        self.param_encodings = list()
        self._tensor_names = list()

    def parse_gguf(self):
        """
        Parse the GGUF file and loads parameter tensors information.
        """
        try:
            self.param_info = load_gguf_checkpoint(self.input_model,
                                                   return_tensors=True,
                                                   extract_encodings=True,
                                                   preserve_original_names=True)
        except Exception as e:
            logger.error(f"Failed to Parse GGUF: {e}")
            raise


        self.arch = self.param_info["config"]["model_type"]
        supported_archs = list(MODEL_TYPE_TO_ARCH.keys())
        if self.arch not in supported_archs:
            sys.exit(f"GGUF BUILDER ONLY SUPPORTS THESE ARCHITECTURES:{supported_archs}! Exiting code execution")

        archs_with_packed_tensors = ["phi3"]
        if self.arch in archs_with_packed_tensors:
            self._split_packed_tensors()
        self._update_lm_head_encodings()
        self._tensor_names = list(self.param_info["tensors"].keys())

    def export_dequantized_weights(self):
        """
        Export de-quantized weights to a GGUF file.
        """
        dequantized_weights_path = os.path.join(self.output_dir, self.filename_prefix + "_dequantized.gguf")
        writer = gguf.GGUFWriter(dequantized_weights_path, self.arch)

        try:
            metadata_info = self.param_info["ignore"]
            config_info = self.param_info["config"]
            tokenizer_info = self.param_info["tokenizer"]
            tokenizer_config_info = self.param_info["tokenizer_config"]

            # add metadata
            writer.add_file_type(metadata_info["file_type"])
            writer.add_quantization_version(metadata_info["quantization_version"])
            writer.add_name(config_info["_model_name_or_path"])

            # add model config
            writer.add_block_count(config_info["num_hidden_layers"])
            writer.add_context_length(config_info["max_position_embeddings"])
            writer.add_embedding_length(config_info["hidden_size"])
            writer.add_feed_forward_length(config_info["intermediate_size"])
            writer.add_head_count(config_info["num_attention_heads"])
            writer.add_head_count_kv(config_info["num_key_value_heads"])
            writer.add_layer_norm_rms_eps(config_info["rms_norm_eps"])

            if config_info.get("vocab_size", None):
                writer.add_vocab_size(config_info["vocab_size"])

            if config_info.get("rope_theta", None):
                writer.add_rope_freq_base(config_info["rope_theta"])

            # add tokenizer_config
            writer.add_tokenizer_model(tokenizer_info["tokenizer_type"])
            writer.add_token_list(tokenizer_info["tokens"])
            writer.add_token_types(tokenizer_info["token_type"])
            writer.add_bos_token_id(tokenizer_info["bos_token_id"])
            writer.add_eos_token_id(tokenizer_info["eos_token_id"])
            if tokenizer_info.get("scores", None):
                writer.add_token_scores(tokenizer_info["scores"])

            if tokenizer_info.get("merges", None):
                writer.add_token_merges(tokenizer_info["merges"])

            if tokenizer_info.get("pad_token_id", None):
                writer.add_pad_token_id(tokenizer_info["pad_token_id"])

            if tokenizer_info.get("unk_token_id", None):
                writer.add_unk_token_id(tokenizer_info["unk_token_id"])

            if config_info.get("original_max_position_embeddings", None):
                writer.add_rope_scaling_orig_ctx_len(config_info["original_max_position_embeddings"])

            if tokenizer_config_info.get("chat_template", None):
                writer.add_chat_template(tokenizer_config_info["chat_template"])

            # add tensors
            for tensor_name, tensor in self.param_info["tensors"].items():
                if self.arch in ["llama", "mistral"]:
                    if ".attn_q." in tensor_name:
                        tensor = permute_weights(tensor,
                                                 config_info["num_attention_heads"],
                                                 config_info["num_attention_heads"])
                    elif ".attn_k." in tensor_name:
                        tensor = permute_weights(tensor,
                                                 config_info["num_attention_heads"],
                                                 config_info["num_key_value_heads"])

                writer.add_tensor(tensor_name, tensor)

            # write to gguf
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        except Exception as e:
            print(f"Error while writing GGUF: {e}")
            traceback.print_exc()
            raise
        finally:
            writer.close()
            # remove param info from memory except encodings
            enc_map = self.param_info.get("encodings_map")
            self.param_info = {"encodings_map": enc_map}
            gc.collect()
        return dequantized_weights_path


    def _split_packed_tensors(self):
            """
            Split packed tensors and their corresponding encodings
            """
            ggml_type_to_string = {type_val: type_string for type_string, type_val in GGML_TYPES.items()}
            arch_config = self.param_info["config"]
            tensor_map = self.param_info["tensors"]
            encodings_map = self.param_info["encodings_map"]
            tensor_to_ggml_string = {tensor_name: ggml_type_to_string[dtype_int] for tensor_name, dtype_int in self.param_info["ggml_types"].items()}

            tensors_to_add = {}
            tensors_to_remove = []
            encodings_to_add = {}
            encodings_to_remove = []

            for tensor_name, encoding in encodings_map.items():

                tensor = tensor_map[tensor_name]

                # split packed qkv
                if "attn_qkv" in tensor_name:

                    logger.debug(f"Splitting packed QKV tensor and encodings for: {tensor_name}")

                    num_heads = arch_config["num_attention_heads"]
                    num_kv_heads = arch_config.get("num_key_value_heads", num_heads)
                    hidden_size = arch_config["hidden_size"]
                    head_dim = arch_config.get("head_dim", hidden_size // num_heads)
                    block_size = GGUF_TYPE_BLOCK_SIZE[tensor_to_ggml_string[tensor_name]]

                    nq = head_dim * num_heads
                    nk = head_dim * num_kv_heads
                    nv = head_dim * num_kv_heads

                    num_blocks_q = (nq * hidden_size) // block_size
                    num_blocks_k = (nk * hidden_size) // block_size
                    num_blocks_v = (nv * hidden_size) // block_size

                    # Split tensors
                    # Q, K, and V are concatenated along axis 0 (the output‑channel axis).
                    # So we slice along axis 0 to separate them.
                    tensor_q = np.copy(tensor[: nq, :])
                    tensor_k = np.copy(tensor[nq: nq + nk, :])
                    tensor_v = np.copy(tensor[nq + nk: nq + nk + nv, :])

                    q_name = tensor_name.replace("attn_qkv", "attn_q")
                    k_name = tensor_name.replace("attn_qkv", "attn_k")
                    v_name = tensor_name.replace("attn_qkv", "attn_v")

                    logger.debug(f"Created split tensors: {q_name}, {k_name}, {v_name}")

                    tensors_to_add[q_name] = tensor_q
                    tensors_to_add[k_name] = tensor_k
                    tensors_to_add[v_name] = tensor_v
                    tensors_to_remove.append(tensor_name)

                    if "bias" not in tensor_name:
                        scales = encoding["scale"]
                        offsets = encoding["offset"]

                        # Split the encodings
                        scales_q = scales[: num_blocks_q]
                        scales_k = scales[num_blocks_q: num_blocks_q + num_blocks_k]
                        scales_v = scales[num_blocks_q + num_blocks_k: num_blocks_q + num_blocks_k + num_blocks_v]

                        offsets_q = offsets[: num_blocks_q]
                        offsets_k = offsets[num_blocks_q: num_blocks_q + num_blocks_k]
                        offsets_v = offsets[num_blocks_q + num_blocks_k: num_blocks_q + num_blocks_k + num_blocks_v]

                        # Create new encoding entries for Q, K, V
                        base_encoding = copy.deepcopy(encoding)

                        q_encoding = copy.deepcopy(base_encoding)
                        q_encoding["name"] = q_name
                        q_encoding["scale"] = scales_q
                        q_encoding["offset"] = offsets_q

                        encodings_to_add[q_name] = q_encoding

                        k_encoding = copy.deepcopy(base_encoding)
                        k_encoding["name"] = k_name
                        k_encoding["scale"] = scales_k
                        k_encoding["offset"] = offsets_k

                        encodings_to_add[k_name] = k_encoding

                        v_encoding = copy.deepcopy(base_encoding)
                        v_encoding["name"] = v_name
                        v_encoding["scale"] = scales_v
                        v_encoding["offset"] = offsets_v

                        encodings_to_add[v_name] = v_encoding
                        encodings_to_remove.append(tensor_name)

                        logger.debug(f"Created split encodings: {q_name}, {k_name}, {v_name}")

                # split packed ffn_up
                if "ffn_up" in tensor_name and tensor_map[tensor_name].shape[0] != arch_config["intermediate_size"]:

                    intermediate_size = arch_config.get("intermediate_size")
                    block_size = GGUF_TYPE_BLOCK_SIZE[tensor_to_ggml_string[tensor_name]]

                    logger.debug(f"Splitting packed FFN_UP tensor and encodings for: {tensor_name}")

                    # Calculate blocks for gate and up
                    tensor = tensor_map[tensor_name]
                    input_size = tensor.shape[1]
                    gate_size = input_size * intermediate_size
                    up_size = input_size * intermediate_size

                    num_blocks_gate = gate_size  // block_size
                    num_blocks_up = up_size // block_size

                    gate_tensor = np.copy(tensor[: intermediate_size, :])
                    up_tensor = np.copy(tensor[intermediate_size: , :])

                    gate_name = tensor_name.replace("ffn_up", "ffn_gate")

                    tensors_to_add[gate_name] = gate_tensor
                    tensors_to_add[tensor_name] = up_tensor

                    logger.debug(f"Created split tensors: {gate_name}, {tensor_name}")

                    if "bias" not in tensor_name:
                        # Split scale and offset arrays
                        scales = encoding["scale"]
                        offsets = encoding["offset"]

                        scales_gate = scales[: num_blocks_gate]
                        scales_up = scales[num_blocks_gate:]

                        offsets_gate = offsets[: num_blocks_gate]
                        offsets_up = offsets[num_blocks_gate:]

                        # Create new encoding entries
                        base_encoding = copy.deepcopy(encoding)

                        # Gate encoding
                        gate_encoding = copy.deepcopy(base_encoding)
                        gate_encoding["name"] = gate_name
                        gate_encoding["scale"] = scales_gate
                        gate_encoding["offset"] = offsets_gate

                        encodings_to_add[gate_name] = gate_encoding


                        # Up encoding
                        encoding["scale"] = scales_up
                        encoding["offset"] = offsets_up

                        logger.debug(f"Created split encodings: {gate_name}, {tensor_name}")

            # Apply changes to encodings_map
            for name in encodings_to_remove:
                if name in encodings_map:
                    del encodings_map[name]
                    logger.debug(f"Removed packed encoding: {name}")

            for name in tensors_to_remove:
                if name in tensor_map:
                    del tensor_map[name]
                    logger.debug(f"Removed packed tensor: {name}")

            for name, encoding in encodings_to_add.items():
                encodings_map[name] = encoding
                logger.debug(f"Added split encoding: {name}")

            for name, tensor in tensors_to_add.items():
                tensor_map[name] = tensor
                logger.debug(f"Added split tensor: {name}")


    def _update_lm_head_encodings(self):
        """
        Copy embedding encodings to lm head encodings, if they are shared.
        """

        embedding_weights_name = GGUF_TENSOR_NAME_STRINGS["embedding_weights_name"]
        lm_head_weights_name = GGUF_TENSOR_NAME_STRINGS["lm_head_weights_name"]

        if embedding_weights_name in self._tensor_names and lm_head_weights_name not in self._tensor_names:
            if embedding_weights_name in self.param_info["encodings_map"]:
                self.param_info["encodings_map"][lm_head_weights_name] = copy.deepcopy(self.param_info["encodings_map"][embedding_weights_name])
                self.param_info["encodings_map"][lm_head_weights_name]["name"] = lm_head_weights_name

    def _generate_param_encodings(self):
        """
        Generate parameter encodings from the parsed GGUF.
        """
        if not self._tensor_names:
            raise RuntimeError("No tensor information available; cannot generate parameter encodings.")

        tensors_to_skip = ['token_embd.weight', 'model.embedding.weight']
        for tensor, encodings in tqdm(self.param_info["encodings_map"].items(), desc="Generating Encodings"):
            if tensor in tensors_to_skip:
                continue
            else:
                self.param_encodings.append(encodings)

    def export_encodings(self):
        """
        Export parameter encodings to a JSON file.
        """

        encoding_file_path = os.path.join(self.output_dir, self.filename_prefix + ".encodings")

        with open(encoding_file_path, "w") as fp_json:
            fp_json.write("{\n")
            fp_json.write("\t\"activation_encodings\": [],\n")
            fp_json.write("\t\"excluded_layers\": [],\n")
            fp_json.write("\t\"param_encodings\": [\n")
            for i, item in enumerate(tqdm(self.param_encodings, desc="Writing Encodings to JSON")):
                json.dump(item, fp_json, sort_keys=True, indent=4)
                if i < len(self.param_encodings) - 1:
                    fp_json.write(",\n")

            fp_json.write("\n],\n")
            fp_json.write("\t\"quantizer_args\": {},\n")
            fp_json.write("\t\"version\": \"1.0.0\"\n")
            fp_json.write("}")

        return encoding_file_path
