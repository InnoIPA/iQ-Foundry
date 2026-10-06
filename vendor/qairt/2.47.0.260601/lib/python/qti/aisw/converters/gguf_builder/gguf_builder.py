# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import re
import sys
import json
import logging
import functools

from typing import Optional
from shutil import copyfile
from transformers.integrations.ggml import convert_gguf_tokenizer
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from . import (GGUFParser, GraphBuilder, update_encodings, SUPPORTED_GGUF_TYPES,
               MODEL_TYPE_TO_ARCH, MODEL_TYPE_TO_TOKENIZER)

logger = logging.getLogger(__name__)


class GGUFBuilder:
    """
    Class to build GenAI model from GGUF file.
    """
    def __init__(self, input_model: str, config_file: Optional[str] = None, output_dir: Optional[str] = None,
                 batch_size: Optional[int] = None, arn_value: Optional[int] = None):
        """
        Constructor
        :param input_model: Path to GGUF file.
        :param config_file: Path to file containing configuration for building GenAI model.
                            (the config.json file generated when saving the huggingface model)
        :param output_dir: Path where the GenAI model and its encodings will be saved.
                           All generated artifacts will be organized within a subfolder under this location.
        :param batch_size: Batch size for the model.
        :param arn_value: Max number of autoregressive tokens at a time.
        """
        self.input_model = os.path.abspath(input_model)
        file_name = os.path.basename(self.input_model)
        self.quant_type = re.search(r'-([^-.]+)\.gguf$', file_name).group(1).upper()
        gguf_quant_type = next((value for value in SUPPORTED_GGUF_TYPES if value in self.quant_type), self.quant_type)
        if gguf_quant_type not in SUPPORTED_GGUF_TYPES:
            logger.error("{} Quantization Type is not supported! Supported Types are {}."
                         " Exiting code execution.".format(self.quant_type, SUPPORTED_GGUF_TYPES))
            sys.exit(1)

        self.config_file = config_file
        # If no explicit output_dir is provided, use the directory of the input model.
        if not output_dir:
            self.output_dir = os.path.dirname(self.input_model)
        # If output_dir is a file, use the directory of the file. Use in CLI Stack (qairt-converter.py)
        elif os.path.splitext(output_dir)[1]:
            self.output_dir = os.path.dirname(os.path.abspath(output_dir))
        # If output_dir is a directory, use the directory as is
        else:
            self.output_dir = os.path.abspath(output_dir)

        # Create directory to export all the generated artifacts
        self.output_dir = os.path.join(self.output_dir, f"gguf_artifacts_{file_name.split('.gguf')[0]}")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except PermissionError:
            print(f"Permission denied: Cannot create directory at '{self.output_dir}'.")
        except OSError as e:
            print(f"An OS error occurred while creating directory at '{self.output_dir}': {e}")

        self.batch_size = batch_size if batch_size else 1
        self.arn_value = arn_value if arn_value else 64
        self.num_layers = None
        self.model_type = None

    def _generate_config_from_gguf(self):
        """
        Generates the configuration files (model and tokenizer) from the GGUF file.
        """
        gguf_data = load_gguf_checkpoint(self.input_model)
        self.num_layers = gguf_data["config"]["num_hidden_layers"]
        self.model_type = gguf_data["config"]["model_type"]

        # If config file is not provided, generate model config from input gguf file
        if not self.config_file:
            config_dict = dict()
            config_dict.update(gguf_data["config"])
            config_dict["architectures"] = [MODEL_TYPE_TO_ARCH[self.model_type]]
            config_dict.pop("_model_name_or_path", None)

            if "vocab_size" not in config_dict:
                config_dict["vocab_size"] = len(gguf_data["tokenizer"]["tokens"])

            if gguf_data["config_rope"].get("long_factor", None) and gguf_data["config_rope"].get("short_factor", None):
                config_dict["rope_scaling"] = {"long_factor": gguf_data["config_rope"].get("long_factor"),
                                               "short_factor": gguf_data["config_rope"].get("short_factor"),
                                               "type": "longrope"}

                config_dict["partial_rotary_factor"] = (len(
                    gguf_data["config_rope"].get("long_factor")) * 2) / config_dict.get("head_dim",
                                                                                        config_dict["hidden_size"] //
                                                                                        config_dict[
                                                                                            "num_attention_heads"])

            with open(os.path.join(self.output_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=4, sort_keys=True)

        # Else, copy config file to output path
        else:
            copyfile(self.config_file, os.path.join(self.output_dir, "config.json"))

        # create generation config file
        gen_config_dict = dict()
        gen_config_dict.update(gguf_data["tokenizer_config"])
        gen_config_dict["_from_model_config"] = True
        gen_config_dict.pop("model_type", None)

        with open(os.path.join(self.output_dir, "generation_config.json"), "w") as f:
            json.dump(gen_config_dict, f, indent=4, sort_keys=True)

        # create tokenizer config file


        # Phi3/3.5 and Phi4 both have the same model type but different
        # tokenizer types. Phi3/3.5 uses llama tokenizer but
        # Phi4 uses gpt tokenizer
        tokenizer_type = self.model_type

        if self.model_type == "phi3":
            if gguf_data["tokenizer"]["tokenizer_type"] != "llama":
                tokenizer_type = gguf_data["tokenizer"]["tokenizer_type"]

        tokenizer, additional_options = convert_gguf_tokenizer(tokenizer_type, gguf_data["tokenizer"])
        fast_tokenizer = MODEL_TYPE_TO_TOKENIZER[tokenizer_type](tokenizer_object=tokenizer, **gguf_data["tokenizer_config"])
        fast_tokenizer.save_pretrained(self.output_dir)

    @staticmethod
    def cache_build_from_gguf(func):
        @functools.wraps(func)
        def wrapper(self):
            onnx_model_path = os.path.join(self.output_dir, f"model_{self.quant_type}.onnx")
            encodings_path = os.path.join(self.output_dir, f"model_{self.quant_type}.encodings")

            if os.path.isfile(onnx_model_path) and os.path.isfile(encodings_path):
                config_path = os.path.join(self.output_dir, "config.json")
                tokenizer_path = os.path.join(self.output_dir, "tokenizer.json")

                # Generate missing config or tokenizer
                if not os.path.isfile(config_path) or not os.path.isfile(tokenizer_path):
                    self._generate_config_from_gguf()

                logger.info("Found existing ONNX model and overrides!")
                logger.info("ONNX model already exists at: {}".format(onnx_model_path))
                logger.info("Quantization Overrides already exist at: {}".format(encodings_path))
                return onnx_model_path, encodings_path
            return func(self)
        return wrapper

    @cache_build_from_gguf
    def build_from_gguf(self):
        """
        Build the GenAI model from the GGUF file.
        """
        try:
            self._generate_config_from_gguf()
            filename_prefix = f"model_{self.quant_type}"

            gguf_parser = GGUFParser(self.input_model, filename_prefix, self.output_dir)
            gguf_parser.parse_gguf()
            gguf_parser._generate_param_encodings()
            dequantized_weights_path = gguf_parser.export_dequantized_weights()

            graph_builder = GraphBuilder(dequantized_weights_path, config_path=self.output_dir, output_dir=self.output_dir,
                                         cache_dir=self.output_dir, batch_size=self.batch_size, arn_value=self.arn_value,
                                         filename_prefix=self.quant_type)
            graph_builder.build_genai_model()
            graph_builder.update_onnx_graph()

            update_encodings(gguf_parser.param_encodings, self.num_layers, self.model_type)

            encodings_path = gguf_parser.export_encodings()
            onnx_model_path = graph_builder.onnx_model_path

            logger.info("ONNX model saved at: {}".format(onnx_model_path))
            logger.info("Quantization Overrides saved at: {}".format(encodings_path))
            return onnx_model_path, encodings_path

        except Exception as e:
            logger.error(f"Failed to build from GGUF: {e}")
            raise
