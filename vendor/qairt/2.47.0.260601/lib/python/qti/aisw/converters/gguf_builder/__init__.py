# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from __future__ import annotations

import os
import sys
import types
import importlib
from importlib import util as iutil

_INIT_DIR = os.path.dirname(__file__)


def _load_file_as_module(target_name: str, path: str) -> types.ModuleType:
    """
    Load a file at `path` as module `target_name` and register into sys.modules.
    """
    spec = iutil.spec_from_file_location(target_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {target_name} at {path}")
    mod = iutil.module_from_spec(spec)
    parent = target_name.rpartition(".")[0]
    mod.__package__ = parent if parent else target_name
    spec.loader.exec_module(mod)
    sys.modules[target_name] = mod
    return mod


def patch_modules():
    """
    Replace:
      - transformers.integrations.ggml
      - transformers.modeling_gguf_pytorch_utils

    in transformers package with custom modules
    """

    # Import transformers and look for parent of ggml.py
    transformers_pkg = importlib.import_module("transformers")
    try:
        integrations_pkg = importlib.import_module("transformers.integrations")
    except ModuleNotFoundError:
        msg = "transformers.integrations package not found; cannot attach ggml module without parent."
        print(msg)
        return

    # Load custom modules into the transformers namespace
    custom_ggml = _load_file_as_module("transformers.integrations.ggml",
                                       os.path.join(_INIT_DIR, "ggml.py"))
    custom_gguf_utils= _load_file_as_module("transformers.modeling_gguf_pytorch_utils",
                                            os.path.join(_INIT_DIR, "modeling_gguf_pytorch_utils.py"))

    # Attach to the parent packages
    setattr(integrations_pkg, "ggml", custom_ggml)

    if hasattr(custom_ggml, "load_dequant_gguf_tensor"):
        setattr(integrations_pkg, "load_dequant_gguf_tensor", custom_ggml.load_dequant_gguf_tensor)

    setattr(transformers_pkg, "modeling_gguf_pytorch_utils", custom_gguf_utils)
    importlib.invalidate_caches()


patch_modules()


from .utils import (
    permute_weights,
    update_encodings,
    update_onnx_graph_helper,
    MODEL_TYPE_TO_ARCH,
    MODEL_TYPE_TO_TOKENIZER,
    SUPPORTED_GGUF_TYPES,
    GGUF_TENSOR_NAME_STRINGS,
    GGUF_TYPE_BLOCK_SIZE
)
from .gguf_parser import GGUFParser
from .graph_builder import GraphBuilder
from .gguf_builder import GGUFBuilder
