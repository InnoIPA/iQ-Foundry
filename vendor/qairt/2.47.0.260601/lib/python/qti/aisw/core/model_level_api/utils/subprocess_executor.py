# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import re
import weakref
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from subprocess import TimeoutExpired, run

import numpy as np
import yaml

logger = getLogger(__name__)
default_execute_timeout = 1800

def _format_output(output):
    if output is None or len(output) == 0:
        return []
    return [line.strip() for line in output.split('\n') if line.strip()]


def execute(command, args=None, cwd='.', shell=False, timeout=default_execute_timeout):
    if args is None:
        args = []

    logger.debug(f"Host command: {command} {args}")
    try:
        process = run([command] + args,
                      cwd=cwd,
                      shell=shell,
                      capture_output=True,
                      text=True,
                      timeout=timeout)
        logger.debug(f"Process return code: ({process.returncode})  stdout: ({process.stdout})  "
                     f"stderr: ({process.stderr})")
        return process.returncode, _format_output(process.stdout), _format_output(process.stderr)
    except TimeoutExpired as error:
        logger.error(f"Timeout of {timeout} seconds expired when running the command")
        return -1, _format_output(error.stdout), _format_output(error.stderr)
    except OSError as error:
        return -1, [], _format_output(str(error))


def get_name_input_pairs_from_input_list(input_list_path) -> list[list[list[str | Path]]]:
    input_list_path = Path(input_list_path)

    # TODO: Below check is to be separated out to a separate function, and changes
    # to be done in the modeltools repo accordingly in net_runner/utils.py.
    # To be handled as part of AISW-147078.
    if input_list_path.suffix in [".yaml", ".yml"]:
        return get_name_input_pairs_from_yaml_input(input_list_path)

    name_input_pair_list = []

    with input_list_path.open() as input_list:
        for line in input_list:
            if line.startswith('#') or line.startswith('%'):
                continue
            inference_inputs = line.strip().split()
            inference_inputs_list = []
            for graph_input in inference_inputs:
                input_name, _, input_filepath = graph_input.rpartition(':=')

                # verify the requested file can be found on the filesystem
                input_path = Path(input_filepath)
                if not input_path.is_file():
                    logger.debug(f'Could not find {input_path}, searching relative to input list')
                    input_path = input_list_path.parent.joinpath(input_path)
                    if not input_path.is_file():
                        raise RuntimeError(f'Could not find input file, tried searching  both '
                                           f'{input_filepath} and {input_path}')

                # sanitize input names by replacing all characters other than alphanumerics with '_'
                input_name = re.sub(r'\W+', '_', input_name)

                inference_inputs_list.append([input_name, input_path.resolve()])
            name_input_pair_list.append(inference_inputs_list)

    return name_input_pair_list

def get_name_input_pairs_from_yaml_input(
    input_list_path: str | Path,
) -> list[list[list[str | Path]]]:
    """
    Extracts input tensor names and file paths from a YAML input file.

    Args:
        input_list_path (str | Path): The path to the YAML input file.

    Returns:
        list: A list of pairs, where each pair contains the input tensor name and its corresponding file path.

    Raises:
        ValueError: If the input file is not a YAML file.
    """
    input_list_path = Path(input_list_path)
    name_input_pair_list = []

    if input_list_path.suffix not in [".yaml", ".yml"]:
        raise ValueError(
            f"Invalid file type. Expected YAML file, got {input_list_path.suffix}"
        )

    with input_list_path.open() as f:
        try:
            yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file: {e}")

    for graph in yaml_data["graphs"]:
        for tensor_set in graph["tensorSets"]:
            current_inference_run_inputs = []  # This list will hold all [name, path] pairs for this specific inference run

            for input_tensor in tensor_set["inputTensors"]:
                input_name = input_tensor["name"]
                input_filepath = input_tensor["filePath"]
                input_path = Path(input_filepath)
                if not input_path.is_file():
                    logger.debug(
                        f"Could not find {input_path}, searching relative to input list"
                    )
                    input_path = input_list_path.parent.joinpath(input_path)
                if not input_path.is_file():
                    raise RuntimeError(
                        f"Could not find input file, tried searching  both "
                        f"{input_filepath} and {input_path}"
                    )
                input_name = re.sub(r"\W+", "_", input_name)

                # Adding the [input_name, input_path] pair to the list for the current inference run
                current_inference_run_inputs.append([input_name, input_path.resolve()])

            # After processing all input tensors for this 'tensor_set', we'll add the collected
            # list of inputs for this inference run to the overall list of all inference runs.
            name_input_pair_list.append(current_inference_run_inputs)

    return name_input_pair_list

class UniqueNameGenerator:
    _counter = defaultdict(int)

    @classmethod
    def generate_unique_name(cls, name):
        cls._counter[name] += 1
        return name + '_' + str(cls._counter[name])


def _process_np_array(input_data,
                      temp_directory,
                      device_directory,
                      input_list_path):
    # input_data is a np.array, so write a single input
    unique_name = UniqueNameGenerator.generate_unique_name('input')
    raw_name = unique_name + '.raw'
    input_path = Path(temp_directory, raw_name)
    with input_path.open('wb') as file:
        input_data.tofile(file)
    with input_list_path.open('a') as file:
        if device_directory is None:
            file.write(str(input_path))
        else:
            if device_directory.startswith('/'):
                file.write(str(device_directory) + '/' + raw_name)
            else:
                file.write(str(device_directory) + '\\' + raw_name)

        file.write('\n')

    return str(input_path)


def _process_dict(input_data,
                  temp_directory,
                  device_directory,
                  input_list_path):
    input_files = []

    # input_data is a dict[str, np.ndarray], so interpret each entry in the dict as a graph input,
    # embedding its name (the dict key) in the input list so it is associated with the proper tensor
    for input_name, input_np_arr in input_data.items():
        # generate file names and write np arrays to disk
        unique_name = UniqueNameGenerator.generate_unique_name(input_name)
        raw_name = unique_name + '.raw'
        input_path = Path(temp_directory, raw_name)
        input_files.append(str(input_path))
        with input_path.open('wb') as file:
            input_np_arr.tofile(file)

        # append to input_list file with generated file and input names
        with input_list_path.open('a') as file:
            if device_directory is None:
                input_str = input_name + ':=' + str(input_path) + ' '
            else:
                if device_directory.startswith('/'):
                    input_str = input_name + ':=' + device_directory + '/' + raw_name + ' '
                else:
                    input_str = input_name + ':=' + device_directory + '\\' + raw_name + ' '
            file.write(input_str)

    # append a newline after all inputs for a single inference are written
    if input_data:
        with input_list_path.open('a') as file:
            file.write('\n')

    return input_files


def _process_input_list(user_input_list_path, device_directory, input_list_path):
    input_files = []

    # all inputs are pushed into the same directory on device, so to prevent overwriting inputs
    # that have the same filename but are separate files on the host, track seen filenames and a
    # mapping of renamed files so unique files with the same filename are not overwritten
    seen_input_filenames = set()
    renamed_input_map = {}

    name_input_pairs = get_name_input_pairs_from_input_list(user_input_list_path)

    with input_list_path.open('w') as file:
        for inference_input in name_input_pairs:
            input_list_line = ''
            for graph_input in inference_input:
                input_name, input_path = graph_input[0], graph_input[1]

                renamed_input_filename = ''
                # Only check and rename the input file if it is going to be pushed to a remote
                # device where it could potentially overwrite an existing file with the same name.
                # If we are running on host, the file is already on the filesystem and can be
                # referred to directly in the input list since it can be anywhere on the filesystem
                if device_directory:
                    input_filename = input_path.name
                    if input_filename in seen_input_filenames:
                        # this filename has been encountered already, determine if it has been seen
                        # before, if not, generate a unique name and insert it into the map
                        if str(input_path) not in renamed_input_map:
                            unique_name = UniqueNameGenerator.generate_unique_name(input_path.stem)\
                                              + input_path.suffix
                            renamed_input_map[str(input_path)] = unique_name
                        renamed_input_filename = renamed_input_map[str(input_path)]
                    else:
                        seen_input_filenames.add(input_filename)

                input_file_str = str(input_path)
                if renamed_input_filename:
                    input_file_str += ';' + renamed_input_filename
                input_files.append(input_file_str)

                input_name_str = input_name + ':=' if input_name else ''
                dst_filename = renamed_input_filename if renamed_input_filename else input_path.name

                if device_directory:
                    if device_directory.startswith('/'):
                        input_file_path_str = str(device_directory) + '/' + dst_filename
                    else:
                        input_file_path_str = str(device_directory) + '\\' + dst_filename

                else:
                    input_file_path_str = str(input_path.parent.joinpath(dst_filename))

                input_list_line += input_name_str + input_file_path_str + ' '

            file.write(input_list_line + '\n')

    return input_files


def generate_input_list(input_data, temp_directory, device_directory=None):
    input_list_path = Path(temp_directory, 'input_list.txt')
    input_files = []

    if isinstance(input_data, np.ndarray):
        raw_name = _process_np_array(input_data,
                                     temp_directory,
                                     device_directory,
                                     input_list_path)
        input_files.append(raw_name)

    elif isinstance(input_data, dict):
        raw_names = _process_dict(input_data,
                                  temp_directory,
                                  device_directory,
                                  input_list_path)
        input_files.extend(raw_names)

    elif isinstance(input_data, list):
        for entry in input_data:
            if isinstance(entry, np.ndarray):
                raw_name = _process_np_array(entry,
                                             temp_directory,
                                             device_directory,
                                             input_list_path)
                input_files.append(raw_name)
            elif isinstance(entry, dict):
                raw_names = _process_dict(entry,
                                          temp_directory,
                                          device_directory,
                                          input_list_path)
                input_files.extend(raw_names)
            else:
                raise ValueError("Expected a list of np.array or dict[str, np.ndarray]")

    elif isinstance(input_data, str) or isinstance(input_data, Path):
        # in case input_data is a Path already, stringify when creating a new Path
        user_input_list_path = Path(str(input_data))
        raw_names = _process_input_list(user_input_list_path, device_directory, input_list_path)
        input_files.extend(raw_names)

    else:
        raise ValueError("Expected np.array or dict[str, np.ndarray] or a list of them or an input"
                         " list file as input")

    return str(input_list_path), input_files


def generate_config_file(backend, temp_directory, sdk_root, device_directory=None):
    backend_config_str = backend.get_config_json()
    if not backend_config_str:
        return "", []

    # write backend config to file
    backend_json_filename = UniqueNameGenerator.generate_unique_name('backend_json') + '.txt'
    backend_json_filepath = Path(temp_directory, backend_json_filename)
    with backend_json_filepath.open('w') as file:
        file.write(backend_config_str)

    logger.info(f'Creating config file {backend_json_filepath} from backend json:\n'
                f'{backend_config_str}')

    files_to_push = []

    # create shared_library_path and config_file_path arguments for the outer json file
    if device_directory:
        config_file_path = Path(device_directory, backend_json_filename)
        files_to_push.append(str(backend_json_filepath))
    else:
        config_file_path = backend_json_filepath

    backend_extensions_lib = backend.backend_extensions_library
    backend_extensions_lib_filepath = Path(sdk_root, 'lib', backend.target.target_name,
                                           backend_extensions_lib)

    if device_directory:
        shared_library_path = Path(device_directory, backend_extensions_lib)
        files_to_push.append(str(backend_extensions_lib_filepath))
    else:
        shared_library_path = backend_extensions_lib_filepath

    # create outer config json as python dictionary and write to file
    outer_config = {
        "backend_extensions":
        {
            "shared_library_path": str(shared_library_path),
            "config_file_path": str(config_file_path)
        }
    }
    netrun_json_filename = UniqueNameGenerator.generate_unique_name('netrun_json') + '.txt'
    netrun_json_filepath = Path(temp_directory, netrun_json_filename)
    with netrun_json_filepath.open('w') as file:
        json.dump(outer_config, file, indent=2)

    if device_directory:
        config_file_arg = f'--config_file {Path(device_directory,netrun_json_filename)}'
        files_to_push.append(str(netrun_json_filepath))
    else:
        config_file_arg = f'--config_file {netrun_json_filepath}'

    return config_file_arg, files_to_push


def create_op_package_argument(backend, device_directory=None):
    op_package_arguments = []
    op_package_names = []

    op_packages = backend.get_registered_op_packages()
    if not op_packages:
        return ''

    for path, interface_provider, target, cpp_stl_path in op_packages:
        op_package_device_name = str(path.stem)
        op_package_suffix = 1
        while op_package_device_name in op_package_names:
            op_package_device_name = f'{path.stem}_{op_package_suffix}'
            op_package_suffix += 1
        if device_directory:
            op_package_path = device_directory + '/' + op_package_device_name + str(path.suffix)
            backend.target.push(str(path), op_package_path)
            op_package_names.append(op_package_device_name)
            if cpp_stl_path is None:
                possible_cpp_stl_path = path.parent / 'libc++_shared.so'
                if Path(possible_cpp_stl_path).exists():
                    backend.target.push(str(possible_cpp_stl_path), device_directory)
                else:
                    if target == 'HTP':
                        logger.warning(f'Warning: libc++_shared.so is not passed. This may cause an error when running on backends other than HTP')
                    else:
                        raise RuntimeError("Need to pass libc++_shared.so path or have it present in the Op Package directory")
            else:
                backend.target.push(str(cpp_stl_path), device_directory)
        else:
            op_package_path = str(path)
        target_str = f':{target}' if target else ''
        op_package_arg = f'{op_package_path}:{interface_provider}{target_str}'
        op_package_arguments.append(op_package_arg)

    return '--op_packages ' + ','.join(op_package_arguments)

def create_set_output_tensors_argument(config, backend, temp_dir, device_directory=None):
    # Early return if no output tensors specified
    if not config or not config.set_output_tensors:
        return ''

    output_tensor_txt = Path(temp_dir, "output_tensors_name.txt")

    # Write output tensor names to file as comma-separated values for qnn-net-run compatibility
    try:
        with output_tensor_txt.open('w') as file:
            file.write(','.join(config.set_output_tensors))
    except IOError as e:
        error_msg = f'Failed to write output tensor list to {output_tensor_txt}: {e}'
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    if device_directory:
        output_device_path = Path(device_directory) / output_tensor_txt.name
        try:
            backend.target.push(str(output_tensor_txt), str(output_device_path))
        except Exception as e:
            error_msg = f'Failed to push output tensor file to device: {e}'
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        output_path = output_device_path
    else:
        output_path = output_tensor_txt

    return f'--set_output_tensors {output_path} '


def output_dir_to_np_array(output_dir, native_outputs, adapter_list=None):


    metadata_path = Path(output_dir, 'execution_metadata.yaml')
    with metadata_path.open() as f:
        metadata = yaml.safe_load(f)

    # TODO AISW-97372: Allow a user to specify a particular graph name to execute in case there are
    #      multiple, when this is done, look up the graph metadata by name instead of using index 0
    graph_metadata = metadata["graphs"][0]

    base_outputs = get_np_array(output_dir, graph_metadata, native_outputs)

    if not adapter_list:
        return base_outputs

    def update_dir_path(out_dir, index):
        return Path(out_dir, graph_metadata["graph_name"] + '_update' + str(index))

    all_outputs = {"base": base_outputs}

    update_idx = 0
    update_path = update_dir_path(output_dir, update_idx)
    while update_path.is_dir():
        update_outputs = get_np_array(update_path, graph_metadata, native_outputs)
        all_outputs[adapter_list[update_idx]] = update_outputs
        update_idx += 1
        update_path = update_dir_path(output_dir, update_idx)

    return all_outputs


def get_np_array(output_dir,graph_metadata, native_outputs):
    def result_dir_path(out_dir, out_index):
        return Path(out_dir, 'Result_' + str(out_index))
    all_outputs = []
    output_index = 0
    result_dir = result_dir_path(output_dir, output_index)
    while result_dir.is_dir():
        outputs = {}
        for f in result_dir.glob('*.raw'):
            output_name = f.stem
            if native_outputs:
                # remove '_native' suffix from output name, qnn-net-run adds this suffix to indicate
                # to users that a .raw file is of a native datatype, but in python we can indicate
                # this via the numpy datatype, so the names should match regardless of datatype
                native_suffix = "_native"
                if output_name.endswith(native_suffix):
                    output_name = output_name[:-len(native_suffix)]

            # find the tensor's metadata and look up the shape and datatype
            output_shape = None
            output_dtype = np.float32
            for output_tensor in graph_metadata["output_tensors"]:
                if output_tensor["tensor_name"] == output_name:
                    output_shape = output_tensor["dimensions"]
                    qnn_dtype = output_tensor["datatype"]
                    output_dtype = get_np_dtype_from_qnn_dtype(qnn_dtype)

            # if native outputs were not requested, qnn-net-run will have written the outputs as
            # float32, but the graph metadata indicates the native output data type. Since
            # dequantization will have already occurred if the datatype was quantized, the output is
            # float32 so we can ignore the output tensor's datatype in the graph metadata
            if not native_outputs:
                output_dtype = np.float32

            output_np_arr = np.fromfile(f, dtype=output_dtype)
            if output_shape:
                output_np_arr = output_np_arr.reshape(output_shape)

            outputs[output_name] = output_np_arr
        all_outputs.append(outputs)

        output_index += 1
        result_dir = result_dir_path(output_dir, output_index)

    return all_outputs


def get_np_dtype_from_qnn_dtype(qnn_dtype):
    qnn_dtype_to_np_dtype = {
        "QNN_DATATYPE_INT_8": np.int8,
        "QNN_DATATYPE_INT_16": np.int16,
        "QNN_DATATYPE_INT_32": np.int32,
        "QNN_DATATYPE_INT_64": np.int64,
        "QNN_DATATYPE_UINT_8": np.uint8,
        "QNN_DATATYPE_UINT_16": np.uint16,
        "QNN_DATATYPE_UINT_32": np.uint32,
        "QNN_DATATYPE_UINT_64": np.uint64,
        "QNN_DATATYPE_FLOAT_16": np.float16,
        "QNN_DATATYPE_FLOAT_32": np.float32,
        "QNN_DATATYPE_FLOAT_64": np.float64,
        "QNN_DATATYPE_SFIXED_POINT_8": np.int8,
        "QNN_DATATYPE_SFIXED_POINT_16": np.int16,
        "QNN_DATATYPE_SFIXED_POINT_32": np.int32,
        "QNN_DATATYPE_UFIXED_POINT_8": np.uint8,
        "QNN_DATATYPE_UFIXED_POINT_16": np.uint16,
        "QNN_DATATYPE_UFIXED_POINT_32": np.uint32,
        "QNN_DATATYPE_BOOL_8": np.uint8
    }

    if qnn_dtype not in qnn_dtype_to_np_dtype:
        raise RuntimeError(f"Cannot convert {qnn_dtype} to np datatype, datatype not supported")

    return qnn_dtype_to_np_dtype[qnn_dtype]


def update_run_config_native_inputs(run_config, input_data):
    if input_data is None:
        return

    native_inputs_required = False
    native_input_tensor_names = []

    # if a user provides a list of inputs (i.e. wants to run multiple inferences of a graph),
    # determine which inputs' datatypes are native based on the first entry in the list. qnn-net-run
    # does not support alternating native and non-native data for a given input between inferences
    if isinstance(input_data, list):
        input_data = input_data[0]

    if isinstance(input_data, np.ndarray):
        if input_data.dtype != np.float32:
            native_inputs_required = True

    elif isinstance(input_data, dict):
        native_names = []
        for input_name, input_arr in input_data.items():
            if input_arr.dtype != np.float32:
                native_inputs_required = True
                native_names.append(input_name)

        # if some inputs are float32, we must explicitly list the native inputs by name so the
        # float32 inputs are still pre-processed by qnn-net-run if needed
        if native_inputs_required and len(native_names) != len(input_data):
            native_input_tensor_names.extend(native_names)

    # if inference input is given as a list of np arrays, the graph has multiple inputs. If a user
    # wants to mix float32 and non-float32 inputs, we need to know the tensor names of all
    # non-float32 inputs to populate the qnn-net-run --native_input_tensor_names argument. So if
    # a mix of float32 and non-float32 np arrays are present in a list of inference inputs AND
    # the user did not provide the native_input_tensor_names config, raise an exception with
    # instructions on how to provide the name mapping
    elif isinstance(input_data, list):
        fp32_seen = False
        native_seen = False
        for np_arr in input_data:
            if np_arr.dtype == np.float32:
                fp32_seen = True
            else:
                native_seen = True
        if fp32_seen and native_seen:
            if not run_config.native_input_tensor_names:
                raise RuntimeError("To mix float32 and non-float32 inputs in an inference, either "
                                   "use a dict[str, np.ndarray] instead of a list[np.ndarray] so "
                                   "the native input tensor names can be inferred, or explicitly "
                                   "provide a list of native input tensor names via the "
                                   "InferenceConfig.native_input_tensor_names config.")

        # if only native types are used, update the config to specify all inputs provided are native
        if native_seen and not fp32_seen:
            native_inputs_required = True

    if native_inputs_required:
        if native_input_tensor_names:
            run_config.native_input_tensor_names = native_input_tensor_names
        else:
            run_config.use_native_input_data = True
