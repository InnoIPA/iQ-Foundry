# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
OnnxModel class - Backward Compatibility Layer

This module provides backward compatibility for the original OnnxModel class.
The implementation now uses GraphContext as the underlying data structure.

DEPRECATED: This module is deprecated. Please use:
    from qairt.optimizer.onnx.graph import GraphContext
    And the passes from qairt.optimizer.onnx.passes directly.

This compatibility layer will be maintained for existing code.

Encapsulate an ONNX model and its associated encodings in a wrapper OnnxModel class to
1. Run transformations
2. Split the onnx model
3. Export the model after running transformations

Example 1 - MHA2SHA

    from qti.aisw.tools.core.utilities.framework.onnx import OnnxModel

    onnx_model = OnnxModel.load(model_path="model.onnx", encodings_path="model.encodings")
    onnx_model.mha2sha_v2()
    onnx_model.export(path="/path/to/export/dir", prefix="model_sha")


Example 2 - Split API

    from qti.aisw.tools.core.utilities.framework.onnx import OnnxModel

    onnx_model = OnnxModel.load(model_path="model.onnx", encodings_path="model.encodings")
    splits = onnx_model.split(
        num_splits=3,
        split_embedding=False,
        split_lm_head=False,
    )

    for idx, _split in enumerate(splits):
        _split.export(path="/path/to/export/dir", prefix=f"model_sha_{idx+1}_of_{len(splits)}")
"""

import json
import pathlib
import os
import tempfile
import warnings
from typing import Any, Dict

import onnx
import onnx_ir as ir

from qairt.optimizer.onnx.graph import ExportedFiles, ExportedUseCase, GraphContext
from qairt.optimizer.onnx.passes import (
    LayoutOptRewriter,
    LoraAlphaExtractor,
    MHA2SHARewriter,
    PermuteKVCacheRewriter,
    ProtectIO,
    UnprotectIO,
)
from qairt.optimizer.onnx.passes.layout_opt.simple_layout_opt import SimpleLayoutOptRewriter
from qairt.optimizer.onnx.passes.splitters import LLMSplitter
from qairt.optimizer.onnx.passes.config import (
    PermuteKVCacheConfig,
    M2sStartPoint,
    MHA2SHAConfig,
)
from qairt.optimizer.onnx.validation.ort_accuracy_checker import (
    verify_onnx_with_inputs_list,
    verify_onnx_with_random_inputs,
)
from qairt.optimizer.onnx.utils.encodings import serialize_graph_encodings
from qairt.optimizer.utils.logger import logger

# Issue deprecation warning when module is imported
warnings.warn(
    "qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model is to be deprecated. "
    "Please plan to use passes with qairt.optimizer.onnx.graph.GraphContext or APIs from qairt.optimizer.onnx.api directly. "
    "This compatibility layer will be maintained for existing code.",
    DeprecationWarning,
    stacklevel=2,
)


class OnnxModel:
    """Manages an Onnx model and optionally, associated encodings

    DEPRECATED: This class is to be deprecated. Please plan to use passes with qairt.optimizer.onnx.graph.GraphContext or APIs
    from qairt.optimizer.onnx.api

    Attributes:
        model: ONNX model
        encodings: AIMET encodings dictionary
    """

    def __init__(
        self,
        graph_context: GraphContext,
        **kwargs,
    ) -> None:
        """Initialize an instance of OnnxModel

        Args:
            graph_context: Instance of GraphContext

        Keyword Args:
            log_level (str): Severity of logging level. Default is INFO
        """
        self.graph_context = graph_context
        
        # Store original named_safetensors before any transformations for LoRA metadata tracking
        # This is needed for record_lora_split_metadata_from_tracing to compare MHA vs SHA tensors
        self.original_named_safetensors = self.graph_context.get_safetensors()
        
        # Store LoRA transform metadata generated during mha2sha_v2
        self.lora_transform_metadata = None

        # Issue deprecation warning when OnnxModel is instantiated
        warnings.warn(
            "OnnxModel is to be deprecated. Please use qairt.optimizer.onnx.graph.GraphContext and " \
            "qairt.optimizer.onnx API and passes directly.",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    def model(self) -> onnx.ModelProto:
        """ONNX model"""
        return ir.serde.serialize_model(self.graph_context.model_ir)

    @property
    def encodings(self) -> dict | None:
        """AIMET encodings dictionary"""
        enc = self.graph_context.get_encodings()
        if enc:
            base_enc_name = "base"

            while base_enc_name in enc:
                base_enc_name += "_"

            base_enc_name = base_enc_name[:-1]

            return serialize_graph_encodings(enc[base_enc_name])
        else:
            return None

    @classmethod
    def load(
        cls,
        model_path: str | os.PathLike,
        encodings_path: str | os.PathLike | None = None,
        lora_adapters_path: str | os.PathLike | dict | None = None,
        lora_tensor_names_path: str | os.PathLike | None = None,
        **kwargs,
    ):
        """Load the model and encodings from file and initialize an instance of OnnxModel

        Args:
            model_path: Path to ONNX model
            encodings_path (Optional): Path to AIMET encodings file. Supported versions are v0.6.1, v1.0.0, and v2.0.0
            lora_adapters_path (Optional): Path to lora adapters yaml file (lora_importer_config)

                The schema for the yaml file should be as follows:
                # Start config
                use_case:  # List of use-cases
                    - name:                 <usecase_1/adapter_1 name>
                      lora_weights:         <path to safetensor file for adapter_1>
                      quant_overrides:      <path to AIMET encodings file for adapter_1>
                    - name:                 <usecase_2/adapter_2 name>
                      lora_weights:         <path to safetensor file for adapter_2>
                      quant_overrides:      <path to AIMET encodings file for adapter_2>
                    ...
            lora_tensor_names_path (Optional): Path to .txt file with updatable lora tensor names

        Keyword Args:
            log_level (str): Severity of logging level. Default is INFO

        Returns:
            An instance of the OnnxModel class
        """
        graph_context = GraphContext.from_files(
            model_path, encodings_path, lora_adapters_path, lora_tensor_names_path
        )
        return cls(graph_context)

    def export(
        self,
        path: str | os.PathLike,
        prefix: str = "model",
    ) -> ExportedFiles:
        """Export model artifacts

        Args:
            path: Directory where the artifacts are to be saved
            prefix: Prefix to model and artifact file names. Defaults to "model"
        """
        # Export using GraphContext
        exported_files = self.graph_context.export(path, prefix)
        
        # If LoRA transform metadata was generated during mha2sha_v2, save it
        if self.lora_transform_metadata:
            export_path = pathlib.Path(path)
            lora_transforms_metadata_path = export_path / "transforms_metadata.json"
            with open(lora_transforms_metadata_path, "w") as f:
                f.write(json.dumps(self.lora_transform_metadata, indent=4))
            exported_files.lora_transform_metadata_path = lora_transforms_metadata_path
            logger.debug(f"LoRA transform metadata saved at {lora_transforms_metadata_path.absolute()}")

        return exported_files

    def mha2sha_v2(
        self,
        extract_lorav2_alpha: bool = False,
        permute_kv_cache_io: bool = False,
        key_cache_name_pattern: str = "past_key_(\\d)+_in|past_key_(\\d)+_out",
        value_cache_name_pattern: str = "past_value_(\\d)+_in|past_value_(\\d)+_out",
        m2s_head_split_map: Dict[int, int] | None = None,
        track_lora_transforms: bool = False,
        transforms_metadata: str | None = None,
        enable_validation: bool = False,
        validation_kwargs: Dict[str, Any] | None = None,
        m2s_additional_start_points: list[M2sStartPoint] | None = None,
        enable_experimental_layout_optimization: bool = False,
        **kwargs,
    ):
        """
        Convert Multi-Head Attention(MHA) layers in the model to Single-Head Attentions(SHA), in-place.
        Optionally, modify the encodings to align with the SHA model.

        If no MHA patterns are found, this function is a no-op.
        Using MHA2SHA v2 implementation

        Args:
            extract_lorav2_alpha: Whether to extract LoRAv2 alpha values. Defaults to False.
                                  (Only applicable for lora-v2 model)
            permute_kv_cache_io: Whether to permute key-value cache inputs/outputs.
                                 Defaults to False.
            key_cache_name_pattern: Pattern for key cache tensor names.
            value_cache_name_pattern: Pattern for value cache tensor names.
            m2s_head_split_map: Mapping for splitting multi-head attention to single-head attention.
                                Defaults to None. Key is the head size of mha, value is the corresponding
                                head size of sha.
                                e.g. "{25:1,128:8}" means split head=25 into head=1 and split head=128
                                into head=8.
                                     "{-1:1}" means split any head size into head=1
            track_lora_transforms: If True, enables tracking and recording of LoRA-related
                                   slice transformations. Defaults to False.
            transforms_metadata: Path to an existing metadata JSON file from a previous stage.
                                 If `track_lora_transforms` is True and this path is provided,
                                 transforms will be loaded from this file before adding new ones.
                                 Defaults to None.
            m2s_additional_start_points: List of M2sStartPoint configurations for specifying additional MHA2SHA
                                         conversion starting points. Defaults to None. By default, MHA2SHA starts
                                         at QKV MatMul operations, but this parameter allows conversion to begin
                                         at custom tensor patterns for non-standard attention architectures.

                                         Each M2sStartPoint specifies:
                                         - name_pattern: Regex pattern to match tensor names (required)
                                         - split_axis: Head axis for splitting (required)
                                         - split_map: Optional mapping from input head count to output head count.
                                                     Uses same format as m2s_head_split_map. Defaults to {-1: 1}

                                         See qairt.api.transforms.model_transformer_config.M2sStartPoint for detailed
                                         documentation of this dataclass.
            enable_validation: Whether to verify the generated ONNX model with ONNX Runtime. Defaults to False.
            validation_kwargs: extra arguments for validation, right now we support:
                - input_raw_list_path: Path of raw input list for verification (same format for qairt-quantizer).
                                Defaults to None. If not provided, random inputs will be used for verification.
                - input_raw_base_dir: Base directory for raw input files. Defaults to None.
            enable_experimental_layout_optimization: Enable experimental layout optimization pass. Defaults to False.
                                                     NOTE: This is an EXPERIMENTAL feature intended for specific 
                                                     models only. Use this ONLY if your model shows poor performance 
                                                     or regression in inference time or higher memory usage on device. 
                                                     For most models, the default layout optimization should be sufficient.
        Returns:
            Detailed information of the generated SHA model
        """
        if validation_kwargs is None:
            validation_kwargs = {}
        
        # Extract split_num and ar_n from kwargs
        split_no = kwargs.get('split_no', None)
        ar_n = kwargs.get('ar_n', None)

        with tempfile.TemporaryDirectory(dir=os.environ.get("QAIRT_TMP_DIR", None)) as tmpdirname:
            ws = os.path.join(tmpdirname, "ws_mha2sha")
            os.makedirs(ws, exist_ok=True)

            mha_onnx_file_path = os.path.join(ws, "mha/model.onnx")
            # we firstly save the mha onnx, so that it can be used for comparison later
            if enable_validation:
                self.graph_context.save_onnx(mha_onnx_file_path)

            out_onnx_special_inputs = {}
            optimized_inputs_preprocs = []
            optimized_outputs_postprocs = []

            if extract_lorav2_alpha:
                lora_alpha_extractor = LoraAlphaExtractor()
                lora_alpha_extractor.apply(self.graph_context)
                lora_alpha_np_value = LoraAlphaExtractor.get_lora_alpha_np_value(self.graph_context)
                out_onnx_special_inputs["lora_alpha"] = lora_alpha_np_value

            if permute_kv_cache_io:
                kv_config = PermuteKVCacheConfig(
                    key_cache_name_pattern=key_cache_name_pattern,
                    value_cache_name_pattern=value_cache_name_pattern,
                )
                kv_cache_io_permutor = PermuteKVCacheRewriter(kv_config)
                kv_cache_io_permutor.apply(self.graph_context)
                optimized_inputs_preprocs.append(kv_cache_io_permutor.preproc_inputs)
                optimized_outputs_postprocs.append(kv_cache_io_permutor.postproc_outputs)

            ProtectIO().apply(self.graph_context)

            mha2sha_config = MHA2SHAConfig(
                m2s_head_split_map=m2s_head_split_map,
                m2s_additional_start_points=m2s_additional_start_points,
            )
            MHA2SHARewriter(mha2sha_config).apply(self.graph_context)
            if enable_experimental_layout_optimization:
                logger.warning(
                    "Experimental layout optimization is enabled. This is an EXPERIMENTAL feature intended for "
                    "specific models only. Use this ONLY if your model shows poor performance or regression in "
                    "inference time or higher memory usage on device. This feature may be removed in the future "
                    "if a better, more generalized solution is identified."
                )
                SimpleLayoutOptRewriter().apply(self.graph_context)
            else:
                LayoutOptRewriter().apply(self.graph_context)
            UnprotectIO().apply(self.graph_context)

            if enable_validation:
                input_raw_list_path = validation_kwargs.get("input_raw_list_path", None)
                input_raw_base_dir = validation_kwargs.get("input_raw_base_dir", None)

                sha_onnx_file_path = os.path.join(ws, "sha", os.path.basename(mha_onnx_file_path))
                self.graph_context.save_onnx(sha_onnx_file_path)

                if input_raw_list_path:
                    logger.info(
                        "input_raw_list_path is specified, validating onnx with onnxruntime and the given inputs"
                    )
                    verify_onnx_with_inputs_list(
                        mha_onnx_file_path,
                        sha_onnx_file_path,
                        input_raw_list_path,
                        input_raw_base_dir,
                        optimized_onnx_special_inputs=out_onnx_special_inputs,
                        optimized_inputs_preprocs=optimized_inputs_preprocs,
                        optimized_outputs_postprocs=optimized_outputs_postprocs,
                    )
                else:
                    logger.info(
                        "input_raw_list_path is not specified, validating onnx with onnxruntime and random inputs"
                    )
                    verify_onnx_with_random_inputs(
                        mha_onnx_file_path,
                        sha_onnx_file_path,
                        optimized_special_inputs=out_onnx_special_inputs,
                        optimized_inputs_preprocs=optimized_inputs_preprocs,
                        optimized_outputs_postprocs=optimized_outputs_postprocs,
                    )

        # Set attributes for backward compatibility
        self.tracing_info = {
            # the complete and detailed tracing information
            "tracing_info": self.graph_context.get_tracing_info(),
            # the consolidated tracing information
            "merged_tracing_info": self.graph_context.get_tracing_info(merged=True),
        }

        # set special lora input, including lora alpha values that extracted from the model
        self.special_sha_inputs = out_onnx_special_inputs
        
        # Generate LoRA transform metadata if requested
        if track_lora_transforms:
            sha_named_safetensors = self.graph_context.get_safetensors()
            
            if sha_named_safetensors and self.original_named_safetensors:
                from qairt.optimizer.onnx.utils.record_lora_metadata import (
                    record_lora_split_metadata_from_tracing,
                )
                
                metadata_graph_id = f"mha2sha_graph{'_split' + str(split_no) if split_no is not None else ''}"
                merged_tracing_info = self.graph_context.get_tracing_info(merged=True)
                
                # Call the metadata recording function with original and transformed safetensors
                self.lora_transform_metadata = record_lora_split_metadata_from_tracing(
                    merged_tracing_info=merged_tracing_info,
                    sha_named_safetensors=sha_named_safetensors,
                    mha_named_safetensors=self.original_named_safetensors,
                    metadata_graph_id=metadata_graph_id,
                    transforms_metadata=transforms_metadata,
                    split_no=split_no,
                    ar_n=ar_n,
                )
                
                if self.lora_transform_metadata:
                    logger.debug("Successfully retrieved metadata from record_lora_split_metadata_from_tracing.")
                else:
                    logger.debug("No metadata generated by record_lora_split_metadata_from_tracing.")
            else:
                logger.debug("LoRA transform tracking is disabled. No metadata generated.")

    def split(
        self,
        *,
        num_splits: int,
        split_embedding: bool = False,
        split_lm_head: bool = False,
        skip_verification: bool = True,
        **kwargs,
    ) -> list["OnnxModel"]:
        """Splits the given ONNX model into multiple smaller models

        Args:
            num_splits: The number of splits to be made
            split_embedding: If True, splits the embeddings. Default is False
            split_lm_head: If True, splits the language model head. Default is False
            skip_verification: If True, skips the verification of the split models. Default is True

        Returns:
            A list of split OnnxModel encapsulating objects
        """
        config = LLMSplitter.Config(
            num_splits=num_splits,
            split_embedding=split_embedding,
            split_lm_head=split_lm_head,
        )
        splits = LLMSplitter(config).split(self.graph_context)

        if not skip_verification:
            # Import validate_splits function
            from qairt.optimizer.onnx.passes.splitters.llm_splitter import validate_splits
            validate_splits(
                ir.serde.serialize_model(self.graph_context.model_ir),
                [ir.serde.serialize_model(s.model_ir) for s in splits],
                logger=logger,
            )

        # Wrap each GraphContext in an OnnxModel for backward compatibility
        return [OnnxModel(graph_context) for graph_context in splits]


# Export for backward compatibility
__all__ = ["OnnxModel", "ExportedFiles", "ExportedUseCase"]
