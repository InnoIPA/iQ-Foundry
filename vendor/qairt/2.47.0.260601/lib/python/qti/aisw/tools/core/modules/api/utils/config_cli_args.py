# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.api import QNNCommonConfig
    from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
    from qti.aisw.tools.core.modules.net_runner import InferenceConfig


def map_log_level(log_level: str) -> str:
    """
    Maps CLI-specific log level strings to standardized internal log level representations.

    Args:
        log_level (str): The log level string provided.

    Returns:
        str: The normalized log level string used internally.
             If the input is not recognized, it is returned unchanged.
    """
    log_level_map = {"critical": "error", "warning": "warn", "trace": "debug"}
    return log_level_map.get(log_level, log_level)


def generate_common_cli_args(config: "QNNCommonConfig") -> str:
    """Generates command line arguments from a QNNCommonConfig object. Takes a QNNCommonConfig object, which
    contains common configuration parameters for both the context binary generator and the net runner module,
    and converts these parameters into a string of command line arguments.

    Args:
        config (QNNCommonConfig): Configuration object containing common parameters.

    Returns:
        str: A string of command line arguments generated from the provided config object.
    """
    args = []

    if config.log_level:
        args.append(f"--log_level {map_log_level(config.log_level.lower())}")
    if config.profiling_level:
        args.append(f"--profiling_level {config.profiling_level}")
    if config.profiling_option:
        args.append(f"--profiling_option {config.profiling_option}")

    platform_options_str = config.platform_options
    if isinstance(config.platform_options, dict):
        platform_options_str = ";".join(f"{k}:{v}" for k, v in config.platform_options.items())
    if platform_options_str:
        args.append(f"--platform_options {platform_options_str}")

    return " ".join(args)


def generate_context_bin_cli_args(config: "GenerateConfig") -> str:
    """Generates command line arguments from a GenerateConfig object. Takes a GenerateConfig object, which
    contains common configuration parameters for the context binary generator, and converts these parameters
    into a string of command line arguments.

    Args:
        config (GenerateConfig): Configuration object containing context binary generator parameters

    Returns:
        str: A string of command line arguments generated from the provided config object.
    """
    qnn_common_config = generate_common_cli_args(config)

    args = [qnn_common_config]

    if config.enable_intermediate_outputs:
        args.append("--enable_intermediate_outputs")
    args.append(f"--input_output_tensor_mem_type {config.input_output_tensor_mem_type}")

    return " ".join(args)


def generate_net_runner_cli_args(config: "InferenceConfig", graph_name: Optional[str] = None) -> str:
    """Generates command line arguments from a InferenceConfig object. Takes a InferenceConfig object, which
    contains common configuration parameters for the net runner module, and converts these parameters into a
    string of command line arguments.

    Args:
        config (InferenceConfig): Configuration object containing net runner module parameters.
        graph_name (Optional[str], optional): The name of the graph for single-graph models.
                                            Used to prefix native input tensor names in the
                                            format 'graph_name:tensor_name'. Defaults to None.

    Returns:
        str: A string of command line arguments generated from the provided config object.
    """
    qnn_common_config = generate_common_cli_args(config)

    args = [qnn_common_config]

    if config.batch_multiplier:
        args.append(f"--batch_multiplier {config.batch_multiplier}")
    if config.use_native_output_data:
        args.append("--use_native_output_files")
    if config.use_native_input_data:
        args.append("--use_native_input_files")
    if config.native_input_tensor_names:
        # For single-graph models, prepend graph name to tensor names if not already prefixed
        tensor_names = config.native_input_tensor_names
        if graph_name and len(tensor_names) > 0:
            # Check if tensor names are already prefixed with graph name
            prefixed_names = []
            for tensor_name in tensor_names:
                if ":" not in tensor_name:
                    # Add graph name prefix for single-graph models
                    prefixed_names.append(f"{graph_name}:{tensor_name}")
                else:
                    # Already prefixed (multi-graph case)
                    prefixed_names.append(tensor_name)
            tensor_names = prefixed_names
        args.append(f"--native_input_tensor_names {','.join(tensor_names)}")
    if config.synchronous:
        args.append("--synchronous")
    if config.debug:
        args.append("--debug")
    if config.perf_profile:
        args.append(f"--perf_profile {config.perf_profile}")
    if config.use_mmap:
        args.append(f"--use_mmap {config.use_mmap}")
    if config.num_inferences:
        args.append(f"--num_inferences {config.num_inferences}")
    if config.duration:
        args.append(f"--duration {config.duration}")
    if config.keep_num_outputs:
        args.append(f"--keep_num_outputs {config.keep_num_outputs}")
    if config.binary_updates:
        args.append(f"--binary_updates {config.binary_updates}")

    return " ".join(args)
