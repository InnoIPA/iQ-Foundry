# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import logging
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, FilePath
from qti.aisw.converters.common import modeltools
from qti.aisw.converters.common.converter_ir.op_graph import QuantUpdatableMode
from qti.aisw.converters.common.graph_property_setter import GraphPropertySetter
from qti.aisw.converters.common.utils import converter_utils, validation_utils
from qti.aisw.converters.qnn_backend.ir_to_dlc import DLCBackend
from qti.aisw.converters.common.backend_awareness import BackendInfo
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
    expect_module_compliance,
)
from qti.aisw.tools.core.modules.converter.common import (
    DLCBackendConfig,
)
from qti.aisw.tools.core.modules.converter.constants import LOGLEVEL
from qti.aisw.tools.core.modules.converter.utils import get_framework_extension
from qti.aisw.tools.core.utilities.qairt_logging import QAIRTLogger

from qti.aisw.converters.common.graph_optimizer import GraphOptimizer, OptimizationStage


class SerializerInputConfig(AISWBaseModel):
    """Configuration model for the serializer input."""

    dlc_backend_config: Optional[DLCBackendConfig] = Field(
        default=None, description="Configuration for DLC backend."
    )
    output_dlc: Optional[str] = Field(default=None, description="Output path to store the DLC.")
    framework: str = Field(description="Source framework from which IRgraph generated.")
    optimized_graph: Any = Field(description="Optimized Graph")
    optimizer_args: Dict[str, Any] = Field(description="Optimizer arguments")
    network_specialization: bool = Field(
        default=False, description="Indicates if network specialization flow is enabled."
    )
    lora_weight_list: Optional[FilePath] = Field(
        default=None, description="Path to a file specifying a list of tensor names that should be updatable."
    )
    quant_updatable_mode: Optional[Literal["none", "adapter_only", "all"]] = Field(
        default=None,
        description="Specify whether/for which tensors the quantization encodings change "
        "across use-cases. In none mode, no quantization encodings are updatable. "
        "In adapter_only mode quantization encodings for "
        "only lora/adapter branch (Conv->Mul->Conv) change across use-case, "
        "the base branch quantization encodings remain the same. "
        "In all mode, all quantization encodings are updatable.",
    )


class SerializerOutputConfig(AISWBaseModel):
    """Configuration model for the serializer output."""

    dlc_path: str = Field(default=None, description="Path to output DLC file")


class SerializerModuleSchemaV1(ModuleSchema):
    """Schema definition for the SerializerModule"""

    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    arguments: SerializerInputConfig
    outputs: SerializerOutputConfig
    name: Literal["SerializerModule"] = "SerializerModule"
    path: Path = Path(__file__)
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class QAIRTSerializer(Module):
    """User interface class for Serializer module"""

    _SCHEMA = SerializerModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """Args:
        logger: A logger instance to be used by the Serializer module
        """
        if logger:
            self._logger = QAIRTLogger.get_logger("SerializerLogger", parent_logger=logger)
        else:
            self._logger = QAIRTLogger.get_logger(
                "SerializerLogger",
                level="INFO",
                formatter_val="extended",
                handler_list=["dev_console"],
            )
        self._debug_level = LOGLEVEL.INFO
        self.backend = None
        self.output_path = None

    def _make_tensors_updatable(self, cpp_graph: Any, tensor_names: List[str]):
        """Marks tensors as updatable in the graph."""
        for tensor_name in tensor_names:
            if cpp_graph.has_tensor(tensor_name):
                self._logger.debug("Marking tensor {} as updatable in the graph.".format(tensor_name))
                cpp_tensor = cpp_graph.get_tensor(tensor_name)
                cpp_tensor.set_updateable(True)

    def _execute_lora_flow(self, args: Namespace, optimized_graph: Any, backend_info_obj: Any):
        """Helper function that executes the LoRA flow within the optimize call."""
        self.backend.initialize()
        prepared_optimized_graph = self.backend.prepare_py_graph(optimized_graph)
        lora_tensor_names = prepared_optimized_graph.get_all_updatable_tensors()
        validation_utils.validate_tensor_names_in_graph(
            lora_tensor_names, prepared_optimized_graph, args.lora_weight_list
        )
        cpp_graph = self.backend.get_ir_graph(prepared_optimized_graph)
        prepared_cpp_graph = self.backend.prepare_cpp_graph(prepared_optimized_graph, cpp_graph)

        if lora_tensor_names:
            self._logger.debug("Marking LoRA adapter weights to updatable")
            self._make_tensors_updatable(prepared_cpp_graph, lora_tensor_names)
        else:
            self._logger.info("Input Model is part of LoRA Use case but the Model but has no LoRA Branches")

        args_dict = GraphOptimizer.ArgParser.convert_args(args)
        graph_optimizer = GraphOptimizer(args_dict)
        graph_optimizer.optimize(prepared_cpp_graph, [OptimizationStage.PostLayout])

        if hasattr(args, "quant_updatable_mode") and args.quant_updatable_mode is not None:
            quant_updatable_mode = QuantUpdatableMode(args.quant_updatable_mode)
        else:
            quant_updatable_mode = None

        graph_property_setter = GraphPropertySetter()
        graph_property_setter.set_graph_properties(prepared_cpp_graph, quant_updatable_mode)
        prepared_cpp_graph = self.backend.quantize_cpp_graph(prepared_optimized_graph, prepared_cpp_graph, args, backend_info_obj)
        graph_property_setter.validate_post_quantize_graph(
            cpp_graph=prepared_cpp_graph,
            quant_updatable_mode=quant_updatable_mode
        )

        self.backend.dlc_serializer.serialize(prepared_cpp_graph)

        if (
            hasattr(args, "quant_updatable_mode")
            and args.quant_updatable_mode == "none"
            and not (hasattr(args, "disable_transform_tracking") and args.disable_transform_tracking)
        ):
            lora_metadata_dict = converter_utils.populate_lora_metadata_json_schema(
                prepared_cpp_graph, prepared_optimized_graph.lora_tensor_names
            )
            lora_metadata_binary_obj = json.dumps(lora_metadata_dict, indent=2).encode("utf-8")
            self.backend.dlc_serializer.add_record_from_buffer(
                lora_metadata_binary_obj, modeltools.DlcRecordType.LORA_CONVERTER_METADATA
            )

        self.backend.finish()

    def serialize(
        self,
        config: SerializerInputConfig,
    ) -> SerializerOutputConfig:
        """Serializes the optimized intermediate representation (IR) graph into a deployable format.

        This method prepares the backend serializer with the appropriate configuration and
        serializes the optimized graph. If network specialization is enabled, it handles
        additional logic such as tensor deduplication and backend initialization.

        Args:
            config (SerializerInputConfig): Configuration object containing the optimized graph
                and the network_specialization flag.

        Returns:
            SerializerOutputConfig: Contains the path to the serialized DLC file.

        Raises:
            Exception: If serialization fails due to any internal error, the exception is logged
            and re-raised.

        Notes:
            In the case of network specialization, `finalize_backend()` should be called after
            serialization is completed for all graphs.
        """
        # transform args to converter namespace
        args = _get_namespace_args(config)
        # Add log level to args so that internal libs use the same log level as API.
        args.debug = self._debug_level
        backend_info_obj = BackendInfo.get_instance(args.backend, args.soc_model)
        network_specialization = config.network_specialization

        try:
            if not self.backend:
                self.backend = DLCBackend(args)  # Use class attribute
                if network_specialization:
                    self.backend.initialize()

            if network_specialization:
                # In network specialization flow, we will avoid checking if the shared context
                # static tensors are already present in DLC
                # The enable_tensor_deduplication flag will enable serializer to look for shared context
                # static tensors data in DLC
                enable_tensor_deduplication = getattr(args, "enable_tensor_deduplication", False)

                self.backend.serialize(
                    config.optimized_graph,
                    network_specialization=True,
                    enable_tensor_deduplication=enable_tensor_deduplication,
                    is_qairt = True,
                    backend_info_obj=backend_info_obj,
                )

            # Support LoRA workflow
            elif config.lora_weight_list:
                self._execute_lora_flow(args, config.optimized_graph, backend_info_obj=backend_info_obj)
            else:
                self.backend.save(config.optimized_graph, is_qairt=True, backend_info_obj=backend_info_obj)

            self.output_path = args.output_path

        except Exception as e:
            self._logger.error("IR graph serialization failed: %s", str(e))
            raise e

        return SerializerOutputConfig(dlc_path=args.output_path)

    def finalize_backend(self) -> str:
        # Should be used only in case of Network Specialization
        """Finalizes the backend for the serializer module.

        Returns:
            Path to the final serialized DLC file
        """
        if self.backend:
            self.backend.finish()

        return self.output_path

    @property
    def _schema(self):
        return self._SCHEMA

    def get_logger(self) -> Any:
        """Returns the logger instance.

        Returns:
            Any: logger instance
        """
        return self._logger

    def properties(self) -> Dict[str, Any]:
        """Returns the properties of the schema.

        Returns:
            Dict[str, Any]: schema properties
        """
        return self._schema.model_json_schema()

    def enable_debug(self, debug_level: int) -> Optional[bool]:
        """Sets serializer log level.

        Args:
            debug_level (int): LOGLEVEL.DEBUG enables DEBUG and higher level messages.
                               LOGLEVEL.INFO enables INFO and higher level messages.

        Returns:
            Optional[bool]: 'True' if debugging is enabled else return 'False'.
        """
        if debug_level < LOGLEVEL.INFO:
            return False
        self._debug_level = debug_level
        return True


def get_dlc_backend(config: SerializerInputConfig) -> DLCBackend:
    """Create a DLCBackend from a SerializerInputConfig.

    Args:
        config: Serializer configuration containing DLC backend settings.

    Returns:
        A DLCBackend instance.
    """
    args = _get_namespace_args(config)
    return DLCBackend(args)


def _get_namespace_args(config: SerializerInputConfig) -> Namespace:
    """Extracts DLC backend-related arguments and returns them in a Namespace."""
    # Handle DLCBackendConfig
    dlc_backend_config = config.dlc_backend_config or DLCBackendConfig()

    # Add DLCBackend arguments
    option_dict = {
        "copyright_file": dlc_backend_config.copyright_file,
        "float_bitwidth": str(dlc_backend_config.float_bitwidth),
        "float_bias_bitwidth": dlc_backend_config.float_bias_bitwidth,
        "model_version": dlc_backend_config.model_version,
        "output_path": dlc_backend_config.output_path,
        "package_name": dlc_backend_config.package_name,
        "quantization_overrides": dlc_backend_config.quantization_overrides,
        "export_format": dlc_backend_config.export_format,
        "calc_static_encodings": dlc_backend_config.calc_static_encodings,
        "quantizer_log": dlc_backend_config.quantizer_log,
        "quantizer_log_level": dlc_backend_config.quantizer_log_level,
        "use_quantize_v2": dlc_backend_config.use_quantize_v2,
        "pack_4_bit_weights": dlc_backend_config.pack_4_bit_weights,
        "disable_dynamic_16_bit_weights": dlc_backend_config.disable_dynamic_16_bit_weights,
    }

    # Add lora arguments
    if getattr(config, "lora_weight_list", None) is not None:
        option_dict["lora_weight_list"] = config.lora_weight_list

    if getattr(config, "quant_updatable_mode", None) is not None:
        option_dict["quant_updatable_mode"] = config.quant_updatable_mode

    # Merge optimizer_args
    optimizer_args = getattr(config, "optimizer_args", None)
    if optimizer_args:
        option_dict.update(optimizer_args)

    args = Namespace(**option_dict)

    # If Converter has not provided output_path, then use output_dlc else determine a default path for the DLC
    if args.output_path:
        output_folder, _ = os.path.split(args.output_path)
    else:
        if config.output_dlc:
            args.output_path = config.output_dlc
            output_folder, _ = os.path.split(config.output_dlc)
        else:
            args.output_path = os.path.join(tempfile.gettempdir(), "optimized_model.dlc")
            output_folder = tempfile.gettempdir()

    ext = get_framework_extension(config.framework)
    # TODO: Clean up once dependency on input network is removed.
    args.input_network = os.path.join(output_folder, "input_network" + ext)

    return args
