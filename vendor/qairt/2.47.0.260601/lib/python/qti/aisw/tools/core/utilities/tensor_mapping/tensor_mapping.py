# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import glob
import os
from typing import Optional

from pydantic import DirectoryPath, FilePath, field_validator, model_validator
from qti.aisw.dlc_utils.snpe_dlc_utils import ModelInfo
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.tensor_mapping.utils import Engine


class TensorMapperInputConfig(AISWBaseModel):
    """Configuration class for generating tensor mappings.

    Attributes:
        dlc_path (Optional[FilePath]): Path to the DLC file. Either this or both
        golden_reference_output and inference_output must be provided.
        engine (Engine): The engine to be used, either QNN or SNPE. Default is QNN.
        golden_reference_output (Optional[DirectoryPath]): Directory path for the golden reference
        output.
        inference_output (Optional[DirectoryPath]): Directory path for the inference output.
        dump_json (bool): If True, dump the tensor mapping JSON file. Default is False.
        transform_dlc_node_name(bool): If False, the node name is not transformed
            Note: This field is valid for tensor mapping on dlc. Default is True

    """

    dlc_path: Optional[FilePath] = None
    engine: Engine = Engine.QNN
    golden_reference_output: Optional[DirectoryPath] = None
    inference_output: Optional[DirectoryPath] = None
    dump_json: bool = False
    transform_dlc_node_name: bool = True

    @model_validator(mode="after")
    def check_args(self):
        """Model validator for TensorMapperInputConfig."""
        dlc_path = self.dlc_path
        golden_reference_output = self.golden_reference_output
        inference_output = self.inference_output

        if dlc_path and (golden_reference_output or inference_output):
            raise ValueError(
                "Invalid configuration: Provide either dlc_path or both golden_reference_output"
                + " and inference_output, not both."
            )

        if not dlc_path:
            if not (golden_reference_output and inference_output):
                raise ValueError(
                    'Invalid configuration: When "dlc_path" is not provided,'
                    + ' both "golden_reference_output" and "inference_output" must be specified.'
                )
            if golden_reference_output == inference_output:
                raise ValueError(
                    "Invalid configuration: golden_reference_output and inference_output must be different."
                )

        return self

    @field_validator("engine")
    def check_engine(cls, value):
        """Field validator for engine."""
        if value not in [Engine.QNN, Engine.SNPE]:
            raise ValueError(
                f"Invalid Engine: {value} is not a valid Engine. It should be either"
                + f"{Engine.QNN.value} or {Engine.SNPE.value}."
            )
        # Temporarily disable SNPE engine for now
        if value == Engine.SNPE:
            raise ValueError(f"Invalid Engine: {value} is not supported yet. Please use {Engine.QNN.value}.")
        return value


class TensorMapperOutputConfig(AISWBaseModel):
    """Configuration class for Tensor Mapper Output.

    Attributes:
        tensor_mapping_output (Optional[Dict]):
            Dictionary to store the output of tensor mappings. Defaults to an empty dictionary.

    """

    tensor_mapping_output: dict = {}


class TensorMapper:
    """TensorMapper Class.

    Example:
        >>> tensor_map = TensorMapper()
        >>> tm = tensor_map.run(tensor_mapping_in_config)

        To dump the tensor mapping to a JSON:
        >>> tm.dump_tensor_mapping_to_file(tm.tensor_mapping_output,"/path/to/json")

    """

    @classmethod
    def run_tensor_mapping_on_dlc(cls, tensor_mapping_in_config: TensorMapperInputConfig) -> dict[str, str]:
        """Runs tensor mapping on a DLC model.

        This method takes a `TensorMapperInputConfig` object as input, which contains
        the path to a DLC model file. It extracts the tensor mapping
        information from the model and returns a dictionary where the keys are the node names in dlc
        and the values are the corresponding tensor names from framework

        Args:
            tensor_mapping_in_config (TensorMapperInputConfig): Input configuration
                containing the DLC model path.

        Returns:
            dict: A dictionary containing the tensor mapping information, where each key is a
                node name from dlc and each value is the corresponding tensor name from framework

        """
        model_info = ModelInfo(str(tensor_mapping_in_config.dlc_path))
        trace_dict = {}
        for graph_name in model_info.graph_names:
            trace_info = model_info.extract_framework_trace_info(graph_name)
            for item in trace_info:
                if item[1] == "TENSOR" and item[3] == "TENSOR":
                    if tensor_mapping_in_config.engine == Engine.QNN:
                        if tensor_mapping_in_config.transform_dlc_node_name:
                            trace_dict[Helper.transform_node_names(item[0])] = item[2]
                        else:
                            trace_dict[item[0]] = item[2]

        return trace_dict

    @classmethod
    def run_tensor_mapping_on_directory(
        cls, tensor_mapping_in_config: TensorMapperInputConfig
    ) -> dict[str, str]:
        """Runs tensor mapping on directories.

        This method takes a `TensorMapperInputConfig` object as input,
        which contains the path to the Framework and Inference outputs.
        It extracts the tensor mapping information from the model and returns
        a dictionary where the keys are the transformed node names and the
        values are the corresponding tensor names.

        Args:
            tensor_mapping_in_config (TensorMapperInputConfig): Input configuration
                containing the DLC model path.

        Returns:
            dict: A dictionary containing the tensor mapping information

        """
        framework_output_dict = cls.get_directory_mappings(
            tensor_mapping_in_config.golden_reference_output, is_golden_reference=True
        )
        inference_output_dict = cls.get_directory_mappings(tensor_mapping_in_config.inference_output)

        """
        The logic below is used to map the layer names between
        framework and inference output directories by comparing
        the value of framework with key of inference output where they are transformed.
        """
        trace_dict = {
            inference_output_dict[value]: key
            for key, value in framework_output_dict.items()
            if value in inference_output_dict.keys()
        }
        return trace_dict

    @classmethod
    def get_directory_mappings(
        cls, directory: DirectoryPath = None, is_golden_reference: bool = False
    ) -> dict[str, str]:
        """Returns a dictionary mapping raw file names to transformed node names or vice versa,
        depending on the configuration.
        This method uses glob to find all .raw files recursively in the specified directory.
        It then constructs a dictionary where the keys are the raw file names without extension
        and the values are the transformed node names, or vice versa, depending on the configuration.

        Args:
            directory (DirectoryPath): The directory to search for .raw files.
            is_golden_reference (bool): Whether the directory is a golden reference directory.

        Returns:
            dict: A dictionary mapping raw file names to transformed node names or vice versa.

        """
        raw_files = []
        directory_mapping_dict = {}
        # Use glob to find all .raw files recursively
        raw_files = glob.glob(os.path.join("**", "*.raw"), recursive=True, root_dir=directory)
        # Get the list of .raw files without extension
        raw_file_without_ext = [os.path.splitext(rf)[0] for rf in raw_files]
        # Construct the dictionary of raw file names to transformed node names
        # or vice versa, depending on the configuration
        for file in raw_file_without_ext:
            if is_golden_reference:
                directory_mapping_dict[file] = Helper.transform_node_names(file)
            else:
                directory_mapping_dict[Helper.transform_node_names(file)] = file
        return directory_mapping_dict

    @classmethod
    def run(cls, tensor_mapping_in_config: TensorMapperInputConfig) -> TensorMapperOutputConfig:
        """Runs the tensor mapping utility.

        This method takes a `TensorMapperInputConfig` object as input and
        returns a `TensorMapperOutputConfig` object.
        It generates a tensor mapping by either running tensor mapping on a
        DLC (Deep Learning Container) path or by comparing golden reference output and
        inference output.
        The resulting tensor mapping is stored in a JSON file and returned as part of
        the output config.

        Args:
            tensor_mapping_in_config (TensorMapperInputConfig): Input configuration
            for tensor mapping generation.

        Returns:
            TensorMapperOutputConfig: Output configuration containing the generated
            tensor mapping and the path to the tensor mapping file.

        """
        trace_dict = {}

        # Trace Generation Logic
        if tensor_mapping_in_config.dlc_path:
            trace_dict = cls.run_tensor_mapping_on_dlc(tensor_mapping_in_config)
        else:
            trace_dict = cls.run_tensor_mapping_on_directory(tensor_mapping_in_config)

        tensor_mapping_out_config = TensorMapperOutputConfig(tensor_mapping_output=trace_dict)

        if tensor_mapping_in_config.dump_json:
            Helper.save_to_json_file(
                tensor_mapping_out_config.tensor_mapping_output,
                os.path.join(os.getcwd(), "tensor_mapping.json"),
            )

        return tensor_mapping_out_config
