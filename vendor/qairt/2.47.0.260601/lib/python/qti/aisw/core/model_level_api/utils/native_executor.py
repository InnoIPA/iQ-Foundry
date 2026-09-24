# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from qti.aisw.core.model_level_api.utils import py_net_run
from qti.aisw.core.model_level_api.utils.subprocess_executor import (
    get_name_input_pairs_from_input_list,
    get_np_dtype_from_qnn_dtype,
)

if TYPE_CHECKING:
    from qti.aisw.tools.core.modules.api import QNNCommonConfig, QNNContextConfig

logger = logging.getLogger(__name__)

def run_config_requires_reinit(curr_config, prev_config):
    if curr_config is None and prev_config is None:
        return False
    elif curr_config is None and prev_config is not None:
        #Curr config will take all default values
        if prev_config.debug:
            return True
        if prev_config.set_output_tensors is not None and len(prev_config.set_output_tensors)>0:
            return True
        if prev_config.perf_profile:
            return True
    elif curr_config is not None and prev_config is None:
        if curr_config.debug:
            return True
        if curr_config.set_output_tensors is not None and len(curr_config.set_output_tensors)>0:
            return True
        if curr_config.perf_profile:
            return True
    else:
        if prev_config.debug != curr_config.debug:
            return True
        if prev_config.set_output_tensors != curr_config.set_output_tensors:
            return True
        if curr_config.perf_profile != prev_config.perf_profile:
            return True
    return False


def create_perf_profile_argument(config):
    perf_profile_str_to_enum = {
        "low_balanced": py_net_run.PerfProfile.LOW_BALANCED,
        "balanced": py_net_run.PerfProfile.BALANCED,
        "default": py_net_run.PerfProfile.DEFAULT,
        "high_performance": py_net_run.PerfProfile.HIGH_PERFORMANCE,
        "sustained_high_performance": py_net_run.PerfProfile.SUSTAINED_HIGH_PERFORMANCE,
        "burst": py_net_run.PerfProfile.BURST,
        "extreme_power_saver": py_net_run.PerfProfile.EXTREME_POWER_SAVER,
        "low_power_saver": py_net_run.PerfProfile.LOW_POWER_SAVER,
        "power_saver": py_net_run.PerfProfile.POWER_SAVER,
        "high_power_saver": py_net_run.PerfProfile.HIGH_POWER_SAVER,
        "system_settings": py_net_run.PerfProfile.SYSTEM_SETTINGS,
        "no_user_input": py_net_run.PerfProfile.NO_USER_INPUT,
        "custom": py_net_run.PerfProfile.CUSTOM,
        "invalid": py_net_run.PerfProfile.INVALID
    }

    if config and config.perf_profile:
        if config.perf_profile not in perf_profile_str_to_enum:
            raise KeyError(f'Performance profile level {config.perf_profile} not recognized')
        return perf_profile_str_to_enum[config.perf_profile]
    return py_net_run.PerfProfile.NO_USER_INPUT


def create_set_output_tensors_argument(config):
    if config and config.set_output_tensors:
        return config.set_output_tensors
    return []


def create_debug_argument(config):
    if config and config.debug:
        return True
    return False


def create_enable_intermediate_outputs_argument(config):
    if config and config.enable_intermediate_outputs:
        return True
    return False


def create_input_output_tensor_mem_type_argument(config):
    if config:
        if (config.input_output_tensor_mem_type == "raw" or
        config.input_output_tensor_mem_type == "memhandle"):
            return config.input_output_tensor_mem_type
        else:
            raise KeyError(f'I/O tensor memory type {config.input_output_tensor_mem_type} not recognized')
    else:
        return "raw"

def create_profile_level_argument(config):
    profile_level_str_to_enum = {
        'off': py_net_run.ProfilingLevel.OFF,
        'basic': py_net_run.ProfilingLevel.BASIC,
        'detailed': py_net_run.ProfilingLevel.DETAILED,
        'backend': py_net_run.ProfilingLevel.BACKEND_CUSTOM
    }

    if config and config.profiling_level:
        if config.profiling_level not in profile_level_str_to_enum:
            raise KeyError(f'profiling level {config.profiling_level} not recognized')
        return profile_level_str_to_enum[config.profiling_level]
    return py_net_run.ProfilingLevel.OFF


def create_profile_option_argument(config):
    profile_option_str_to_enum = {
        'none': py_net_run.ProfilingOption.NONE,
        'optrace': py_net_run.ProfilingOption.OPTRACE
    }

    if config and config.profiling_option:
        if config.profiling_option not in profile_option_str_to_enum:
            raise KeyError(f'profiling option {config.profiling_option} not recognized')
        return profile_option_str_to_enum[config.profiling_option]
    return py_net_run.ProfilingOption.NONE


def create_log_level_argument(config):
    log_level_alias_map = {
        'critical': 'error',
        'warning': 'warn',
        'trace': 'debug'
    }

    log_level_str_to_enum = {
        'error': py_net_run.QnnLogLevel.ERROR,
        'warn': py_net_run.QnnLogLevel.WARN,
        'info': py_net_run.QnnLogLevel.INFO,
        'verbose': py_net_run.QnnLogLevel.VERBOSE,
        'debug': py_net_run.QnnLogLevel.DEBUG
    }

    if not config or not config.log_level:
        return py_net_run.QnnLogLevel.ERROR
    log_level = config.log_level.lower()
    log_level = log_level_alias_map.get(log_level, log_level)

    try:
        return log_level_str_to_enum[log_level]
    except KeyError:
        raise KeyError(f'log level {log_level} not recognized')


def create_op_package_argument(backend):
    op_packages = backend.get_registered_op_packages()
    if not op_packages:
        return ''

    op_package_strings = []
    for pkg_path, pkg_provider, pkg_target, _ in op_packages:
        target_str = f':{pkg_target}' if pkg_target else ''
        op_package_str = f'{pkg_path}:{pkg_provider}{target_str}'
        op_package_strings.append(op_package_str)

    return ','.join(op_package_strings)


def create_backend_extension_argument(backend, temp_directory, sdk_root):
    backend_extension_lib_name = backend.backend_extensions_library
    backend_extension_lib_path = ''
    if backend_extension_lib_name:
        backend_extension_lib_path = Path(sdk_root, 'lib', backend.target.target_name,
                                           backend_extension_lib_name)

    backend_config_str = backend.get_config_json()
    if not backend_config_str:
        return [f'{backend_extension_lib_path}', '']

    backend_json_path = Path(temp_directory, 'backend_json.txt')
    with backend_json_path.open('w') as file:
        file.write(backend_config_str)

    return [f'{backend_extension_lib_path}', f'{backend_json_path}']


def create_context_priority_argument(context_config: 'QNNContextConfig'):
    context_priority_str_to_enum = {
        "low": py_net_run.QnnPriority.LOW,
        "normal": py_net_run.QnnPriority.NORMAL,
        "normal_high": py_net_run.QnnPriority.NORMAL_HIGH,
        "high": py_net_run.QnnPriority.HIGH
    }
    context_priority = py_net_run.QnnPriority.DEFAULT
    if context_config and context_config.context_priority:
        context_priority = context_priority_str_to_enum[context_config.context_priority]
    return context_priority


def create_async_execute_queue_depth_argument(context_config: 'QNNContextConfig'):
    async_execute_queue_depth = 0
    configure_async_execute_queue_depth = False
    if context_config and context_config.async_execute_queue_depth:
        async_execute_queue_depth = context_config.async_execute_queue_depth
        configure_async_execute_queue_depth = True
    return async_execute_queue_depth, configure_async_execute_queue_depth


def create_enable_graphs_argument(context_config: 'QNNContextConfig'):
    enable_graphs = []
    if context_config and context_config.enable_graphs:
        enable_graphs = context_config.enable_graphs
    return enable_graphs


def create_memory_limit_argument(context_config: 'QNNContextConfig'):
    memory_limit_hint = 0
    memory_limit_hint_present = False
    if context_config and context_config.memory_limit_hint:
        memory_limit_hint = context_config.memory_limit_hint
        memory_limit_hint_present = True
    return memory_limit_hint, memory_limit_hint_present


def create_persistent_binary_argument(context_config: 'QNNContextConfig'):
    is_persistent_binary = False
    if context_config and context_config.is_persistent_binary:
        is_persistent_binary = context_config.is_persistent_binary
    return is_persistent_binary


def create_cache_compatibility_argument(context_config: 'QNNContextConfig'):
    compatibility_str_to_enum = {
        "permissive": py_net_run.CompatibilityType.PERMISSIVE,
        "strict": py_net_run.CompatibilityType.STRICT
    }
    cache_compatibility_mode = py_net_run.CompatibilityType.UNDEFINED
    cache_compatibility_mode_present = False
    if context_config and context_config.cache_compatibility_mode:
        cache_compatibility_mode = compatibility_str_to_enum[context_config.cache_compatibility_mode]
        cache_compatibility_mode_present = True
    return cache_compatibility_mode, cache_compatibility_mode_present


def create_batch_multiplier_argument(config):
    return config.batch_multiplier if config and config.batch_multiplier else 1


def create_output_datatype_argument(config):
    return 'native_only' if config and config.use_native_output_data else 'float_only'

def create_use_mmap_argument(config):
    if config and config.use_mmap:
        return True
    return False

def create_num_inferences_argument(config):
    return config.num_inferences if config and config.num_inferences else -1

def create_total_duration_argument(config):
    return config.duration if config and config.duration else -1.0

def create_keep_num_outputs_argument(config):
    return config.keep_num_outputs if config and config.keep_num_outputs else -1


def create_platform_options_argument(config: Optional['QNNCommonConfig'] = None) -> str:
    """
    Validates and converts platform options to string.

    Args:
        config (Optional[QNNCommonConfig]): Configuration object that may contain platform options.

    Returns:
        str: Platform options in the format 'key0:value0;key1:value1;key2:value2',
        or an empty string if no platform options are provided.

    Raises:
        ValueError: If the input string is not in the correct format.
        TypeError: If the input is neither a dictionary nor a string.
    """
    if config is None or not config.platform_options:
        return ""

    platform_options = config.platform_options

    if isinstance(platform_options, dict):
        return ';'.join(f'{k}:{v}' for k, v in platform_options.items())
    elif isinstance(platform_options, str):
        try:
            # Attempt to parse the string into a dictionary to ensure it is properly formatted.
            _ = dict(item.split(":") for item in platform_options.split(";"))
            return platform_options
        except ValueError:
            raise ValueError("Platform options string is not in the correct format: "
                             "'key0:value0;key1:value1;key2:value2'")
    else:
        raise TypeError("Platform options must be a dictionary or a string")

def create_adapter_weight_config_file_argument(config: Optional['QNNCommonConfig'] = None) -> str:
    """
    Validates and return adapter weight config path as string.

    Args:
        config (Optional[QNNCommonConfig]): Configuration object that may contain adapter weight config path.

    Returns:
        str: Adapter weight config path,
        or an empty string if no adapter weight config file is provided.
    """
    if config is None or not config.adapter_weight_config_file:
        return ""

    return str(config.adapter_weight_config_file)

def input_list_to_in_memory_input(input_list_path,
                                  native_inputs,
                                  native_input_tensor_names,
                                  input_name_dtype_pairs) -> List[Dict[str, np.ndarray]]:
    """
    Loads input tensors from an input list file into memory. It reads an input list file,
    extracts the input tensor information, and loads the input tensors into memory.

    Args:
        input_list_path (Path): The path to the input list file.
        native_inputs (bool): Whether to use native inputs.
        native_input_tensor_names (list): List of native input tensor names.
        input_name_dtype_pairs (list): List of input name and data type pairs.

    Returns:
        list: A list of dictionaries, where each dictionary contains the tensor name
            and its corresponding NumPy array.

    Notes:
        If `native_inputs` is True, all inputs are assumed to be their native datatype.
        If `native_input_tensor_names` is provided, only the specified inputs are loaded
        as native inputs, and all others are loaded as float32.
    """
    input_data_list = []

    name_input_pairs = get_name_input_pairs_from_input_list(input_list_path)
    input_name_dtype_dict = None
    if input_name_dtype_pairs:
        if isinstance(input_name_dtype_pairs, list):
            input_name_dtype_dict = dict(input_name_dtype_pairs)
        else:
            input_name_dtype_dict = input_name_dtype_pairs

    for inference_input in name_input_pairs:
        inference_input_dict = {}
        for input_idx, (input_name, input_path) in enumerate(inference_input):
            np_dtype = np.float32
            if not input_name:
                input_name = f'placeholder_input_{input_idx}'
                if native_inputs:
                    # if input name is not present in the input list and all inputs are requested
                    # as native, look up the datatype based on index
                    input_dtype = input_name_dtype_pairs[input_idx][1]
                    np_dtype = get_np_dtype_from_qnn_dtype(input_dtype)
                elif native_input_tensor_names:
                    # if only certain inputs are requested, look up the input name by index and see
                    # if it is one of the requested inputs, if so look up the datatype by index
                    input_name = input_name_dtype_pairs[input_idx][0]
                    if input_name in native_input_tensor_names:
                        input_dtype = input_name_dtype_pairs[input_idx][1]
                        np_dtype = get_np_dtype_from_qnn_dtype(input_dtype)
            else:
                # if input name is present, look up the datatype based on name
                if native_inputs or (native_input_tensor_names and
                                     input_name in native_input_tensor_names):
                    input_dtype = input_name_dtype_dict.get(input_name)
                    if input_dtype is not None:
                        np_dtype = get_np_dtype_from_qnn_dtype(input_dtype)

            inference_input_dict[input_name] = np.fromfile(input_path, dtype=np_dtype)
        input_data_list.append(inference_input_dict)

    return input_data_list

# a utility to temporarily change into a working directory, reverting back to the previous working
# directory when the object goes out of scope
@contextmanager
def temporaryDirectoryChange(temp_dir):
    old_cwd = os.getcwd()
    os.chdir(Path(temp_dir).resolve())

    try:
        yield
    finally:
        os.chdir(old_cwd)
