# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import Literal, Optional

from pydantic import Field, FilePath, ConfigDict, model_validator
from qti.aisw.converters.common import backend_info, ir_quantizer
from qti.aisw.tools.core.modules.api import AISWBaseModel


class DLCBackendConfig(AISWBaseModel):  # noqa: D101

    model_config = ConfigDict(arbitrary_types_allowed=True)
    copyright_file: str = Field(
        default="",
        description="Path to copyright file. If provided, "
        "the content of the file will be added "
        "to the output model",
    )
    float_bitwidth: Literal[32, 16] = Field(
        default=32, description="Convert the graph to specified float bitwidth."
    )
    float_bias_bitwidth: Literal[16, 32] = Field(
        default=32, description="Option to select the bit-width to use for float bias tensor."
    )
    model_version: str = Field(
        default="",
        description="User-defined ASCII string to identify the model, only first 64 bytes will be stored.",
    )
    output_path: str = Field(
        default="",
        description="Path where the converted output model should be saved. If"
        " not specified, the converter model will be written to a "
        "file with same name as the input model",
    )
    package_name: str = Field(
        default="",
        description="A global package name to be used for each node in "
        "the Model.cpp file. Defaults to Qnn header defined "
        "package name.",
    )
    quantization_overrides: Optional[FilePath] = Field(
        default=None, description="Option to specify a json file with parameters to use for "
    )
    export_format: Literal["DLC_DEFAULT", "DLC_STRIP_QUANT"] = Field(default="DLC_DEFAULT",
        description="DLC_DEFAULT (default)\n"
        "- Produce a Float graph given a Float Source graph\n"
        "- Produce a Quant graph given a Source graph with provided Encodings\n"
        "DLC_STRIP_QUANT\n"
        "- Produce a Float Graph with discarding Quant data\n")
    calc_static_encodings: bool = Field(
        default=False,
    )
    quantizer_log: str = Field(
        default="",
        description="Valid for use with v2.0.0 JSON schema for quantization "
                    "overrides or when --use_quantize_v2 is provided. "
    )
    quantizer_log_level: ir_quantizer.LogLevel = Field(
        default=ir_quantizer.LogLevel.NONE,
        description="Sets the logging level in the quantizer. See --quantizer_log.\n"
                    "INFO: Emits a file in the CSV format. Requires --quantizer_log <file_name.csv> to be set. "
                    "TRACE: Emits a file in the TXT format. Requires --quantizer_log "
                    "<file_name.txt> to be set. "
                    "NONE: Default value. No file is emitted. "
                    "Warnings and errors are emitted to the console.")
    use_quantize_v2: bool = Field(
        default=False,
        description="Enables IrQuantizerV2 flow."
    )
    pack_4_bit_weights: bool = Field(
        default=False,
    )
    disable_dynamic_16_bit_weights: bool = Field(
        default=False,
    )


class BackendInfoConfig(AISWBaseModel):  # noqa: D101
    backend: str = Field(
        default="",
        description="Option to specify the backend on which the model needs to "
        "run. Providing this option will generate a graph optimized "
        "for the given backend."
        "Options for backend - CPU, DSP, GPU, HTP, HTA, AIC and LPAI",
    )
    soc_model: str = Field(
        default="",
        description="Option to specify the SOC on which the model "
        " needs to run. This can be found from SOC info"
        " of the device and it starts with strings "
        " such as SDM, SM, QCS, IPQ, SA, QC, SC, SXR, "
        " SSG, STP, QRB, or AIC.",
    )

    @model_validator(mode="after")
    def validate_backend(self):
        """Checks if a backend is supported"""
        # TODO: Default for backend should not be an empty string, this needs to be removed
        if self.backend == "":
            return self

        if self.backend not in backend_info.supported_backends():
            raise ValueError("'{}' backend is not supported.".format(self.backend))
        return self
