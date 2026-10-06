# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================
import sys
import multiprocessing as mp
import os
import tempfile
import traceback

from qti.aisw.tools.core.modules.emitter.torch_emitter import (EmitterInputConfig,
                                                               TorchEmitterAndConfigGenerator)
from qti.aisw.tools.core.modules.light_weight_quantizer.lwq_module import (LightWeightQuantizerInputConfig,
                                                                           QAIRTLightWeightQuantizer)
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning, log_error
import qti.aisw.converters.aimet.utils as aimet_utils

from qti.aisw.converters.common import ir_graph as ir_graph_lib
IrGraph = ir_graph_lib.IrGraph

from qti.aisw.emitter.utils.ir_graph_utils import serialize_ir_graph_to_dlc, get_dlc_reader


def prepare_parent_environment_for_aimet(init_method):
    def wrapper(*args, **kwargs):
        init_method(*args, **kwargs)

        aimet_venv_python_path = args[0].aimet_venv_python_path
        aimet_venv_base_dir = os.path.dirname(os.path.dirname(aimet_venv_python_path))

        if sys.platform == "win32":
            site_packages = os.path.join(aimet_venv_base_dir, sys.platlibdir, "site-packages")
        else:
            major_v = sys.version_info.major
            minor_v = sys.version_info.minor
            site_packages = os.path.join(aimet_venv_base_dir, sys.platlibdir, f"python{major_v}.{minor_v}", "site-packages")

        import site
        site.addsitedir(site_packages)

        # Order the python path so that the site-package of the virtual environment is at the beginning
        # of the python path list. This prioritizes the version of a package in the virtual environment
        sys.path = ([path for path in sys.path if aimet_venv_base_dir in path] +
                    [path for path in sys.path if aimet_venv_base_dir not in path])

    return wrapper


def restore_parent_environment_after_aimet(terminate_method):
    def wrapper(*args, **kwargs):
        terminate_method(*args, **kwargs)

        # Restore sys.path, after terminating AIMET process
        sys.path = args[0].parent_sys_path

    return wrapper


def get_python_executable():
    aimet_env_python_exec = os.environ.get("AIMET_ENV_PYTHON")
    error_msg = ('Provided python executable at $AIMET_ENV_PYTHON is invalid. Please run '
                 'aimet_env_setup.sh to ensure AIMET_ENV_PYTHON is set to <aimet_venv>/bin/python')
    if os.path.exists(aimet_env_python_exec):
        # This returns python version, must contain 'python' in version string,
        # if it is a valid python interpreter
        try:
            python_version = os.popen(f'{aimet_env_python_exec} --version').read().strip()
            assert 'python' in python_version.lower(), error_msg
            log_info('Validated environment variable, AIMET_ENV_PYTHON')
            return aimet_env_python_exec
        except Exception:
            raise EnvironmentError(error_msg)
    else:
        raise EnvironmentError(error_msg)


def quantize_model_with_aimet(conn, tmpdir, opts, backend_base_config_path):
    """
    Call this function within a subprocess to execute aimet specific code in a separate virtual environment
    """
    weight_file_path = None
    encoding_file_path = None
    try:
        # Import this only after adding the virtual environment's site package to python path
        from qti.aisw.converters.aimet.qnn_quantsim_adapter import QnnToAimetAdapter

        qnn_adapter = QnnToAimetAdapter(opts, backend_base_config_path, datadir=tmpdir, use_cuda=True)
        if qnn_adapter.is_valid_opts():
            weight_file_path, encoding_file_path = qnn_adapter.generate_weight_and_encoding()
    except Exception as e:
        traceback.print_exc()
    finally:
        # For single process setup, there is no need for IPC
        if conn is None:
            return weight_file_path, encoding_file_path
        else:
            conn.send([weight_file_path, encoding_file_path])
            conn.close()


class AimetProcess(mp.Process):
    @prepare_parent_environment_for_aimet
    def __init__(self, target, args, aimet_venv_python_path, parent_sys_path):
        super(AimetProcess, self).__init__(target=target, args=args)
        self.target = target
        self.args = args
        self.aimet_venv_python_path = aimet_venv_python_path
        self.parent_sys_path = parent_sys_path

    def run(self):
        self.target(*self.args)

    @restore_parent_environment_after_aimet
    def terminate(self):
        super(AimetProcess, self).terminate()


def aimet_quantizer(ir_graph, opts):
    aimet_env_python_exec = get_python_executable()
    if not aimet_env_python_exec:
        raise EnvironmentError(
            """Environment variable 'AIMET_ENV_PYTHON' not set.
            Please run  'source $QNN_SRC/QTI/scripts/aimet_env_setup.sh --env-path <PATH>' if you want to use aimet quantizer
            or omit the '--use_aimet_quantizer' flag to use the default quantizer"""
        )
    # Create a multiprocessing context with start method 'spawn' and set the python executable path
    with tempfile.TemporaryDirectory() as tmpdir:
        unquantized_dlc_filename = 'model_fp'
        unquantized_dlc_path = tmpdir
        serialize_ir_graph_to_dlc(ir_graph, unquantized_dlc_path, unquantized_dlc_filename)
        # Set multiprocessing context to 'spawn' to keep AIMET process be independent of main process environment
        mp.set_start_method("spawn", force=True)
        mp.set_executable(aimet_env_python_exec)
        # Create a process and run aimet-specific code within the context of that process
        parent_conn, child_conn = mp.Pipe()
        fullpath = os.path.join(unquantized_dlc_path, unquantized_dlc_filename + '.dlc')
        backend_base_config_path = opts.torch_emitter_and_config_generator(fullpath, tmpdir)
        process = AimetProcess(target=quantize_model_with_aimet,
                               args=(fullpath, child_conn, tmpdir, opts, backend_base_config_path),
                               aimet_venv_python_path=aimet_env_python_exec,
                               parent_sys_path=sys.path.copy())
        process.start()
        retval = parent_conn.recv()
        weight_file_path = retval[0]
        encoding_path = retval[1]
        process.join()
        process.terminate()

        ##### Get Quantized DLC ######
        if (weight_file_path is not None and os.path.exists(weight_file_path)) and (encoding_path is not None and
                                                                                    os.path.exists(encoding_path)):
            ##### Get Quantized DLC ######
            lwq_input_config = LightWeightQuantizerInputConfig(path=tmpdir,
                                                               filename_prefix=opts.emitter_filename,
                                                               dlc_path=fullpath,
                                                               weight_file_path=weight_file_path,
                                                               encoding_path=encoding_path,
                                                               quantize_dlc=True,
                                                               activation_bitwidth=opts.act_bitwidth,
                                                               float_bias_bitwidth=opts.float_bias_bw)
            lwq_quantizer = QAIRTLightWeightQuantizer()
            lwq_output_config = lwq_quantizer.export(lwq_input_config)
            quantized_dlc_path = lwq_output_config.dlc_path

            if quantized_dlc_path is not None and os.path.exists(quantized_dlc_path):
                reader = get_dlc_reader(quantized_dlc_path)
                return reader
            else:
                log_error('Exception occured in LWQ Module')
                sys.exit()
        else:
            log_error('Exception occured in Spawned AIMET Process, Unable to proceed with Quantization')
            sys.exit()


class AimetQuantizerOpts:
    def __init__(self,
                 input_network,
                 output_path,
                 input_list,
                 quant_schemes,
                 float_fallback,
                 disable_legacy_quant_scheme_opts,
                 algorithms,
                 act_bitwidth,
                 weights_bitwidth,
                 bias_bitwidth,
                 float_bias_bw,
                 percentile_calibration_value,
                 ignore_encodings,
                 use_per_channel_quantization,
                 use_per_row_quantization,
                 use_native_input_files,
                 use_native_output_files,
                 backend_name,
                 config=None):

        # TODO: Remove this once Windows based AIMET build is available
        if sys.platform == "win32":
            raise RuntimeError('AIMETQuantizer is not supported on Windows yet')

        self.input_network = input_network
        self.output_path = output_path
        self.input_list = input_list
        self.quant_schemes = quant_schemes
        self.float_fallback = float_fallback
        # Flag to detect whether to use --act_quantizer and --param_quantizer (or) [--act_quantizer_calibration, --act_quantizer_schema]
        # [--param_quantizer_calibration, --param_quantizer_schema] to resolve AIMET Quant Scheme
        self.disable_legacy_quant_scheme_opts = disable_legacy_quant_scheme_opts
        self.algorithms = algorithms
        self.act_bitwidth = act_bitwidth
        self.weights_bitwidth = weights_bitwidth
        self.bias_bitwidth = bias_bitwidth
        self.float_bias_bw = float_bias_bw
        self.percentile_calibration_value = percentile_calibration_value
        self.ignore_encodings = ignore_encodings
        self.use_per_channel_quantization = use_per_channel_quantization
        self.use_per_row_quantization = use_per_row_quantization
        self.use_native_input_files = use_native_input_files
        self.use_native_output_files = use_native_output_files
        self.backend_name = backend_name
        self.config = config
        self.validate_aimet_quant_opts()
        # In case of high level api, input_network is not getting propagated properly so
        # commenting this out. Once resolved, we can enable this logic and remove the logic to
        # infer the name from output dlc path
        self.emitter_filename, _ = os.path.splitext(os.path.basename(self.output_path))
        # if os.path.isfile(self.input_network):
        #     self.emitter_filename, _ = os.path.splitext(os.path.basename(self.input_network))
        self.emitter_filename = self.emitter_filename + '_prepared_model'
        self.emitter_model_name = aimet_utils.replace_invalid_chars_for_variable(self.emitter_filename + "_model") # Same as filename +'_model' for simplicity

        self.input_shapes = None
        self.input_dtypes = None

    def torch_emitter_and_config_generator(self, dlc_path, datadir):

        emitter_input_config = EmitterInputConfig(input_graph=dlc_path,
                                                  backend_name=self.backend_name,
                                                  path=datadir,
                                                  filename=self.emitter_filename,
                                                  model_name=self.emitter_model_name,
                                                  ignore_encodings=self.ignore_encodings
                                                  )
        emitter = TorchEmitterAndConfigGenerator()
        emitter_out_config = emitter.prepare_model(emitter_input_config)
        self.input_shapes = emitter._get_ir_graph_input_shapes()
        self.input_dtypes = emitter._get_ir_graph_input_dtypes()

        return emitter_out_config.backend_base_config_path

    def validate_aimet_quant_opts(self):
        # TODO: Support --use_native_output_files if required
        if self.use_native_output_files:
            raise Exception("AIMET Quantizer doesn't support --use_native_output_files")

        # TODO: Support --bias_bitwidth 8
        if self.bias_bitwidth != 32:
            # TODO: raise Exception once the default is changed to 32
            log_warning(f"AIMET Quantizer doesn't support {self.bias_bitwidth} for --bias_bitwidth or --bias_bw, using 32")
            self.bias_bitwidth = 32

        if self.config is not None and self.float_fallback:
            raise Exception("Can't provide --config and --float_fallback together")

        if len(self.algorithms) != 0 and self.float_fallback:
            raise Exception("Can't provide --algorithms and --float_fallback together")

        if len(self.algorithms) > 1:
            raise RuntimeError("Currently AIMET Quantizer can't run more than one algorithm!")

        else:
            # When no config is provided, algorithms can take 'cle' or 'adaround' (default case)
            if self.config is None and len(self.algorithms) == 1 and self.algorithms[0] not in ['adaround', 'cle']:
                raise Exception("When no --config is provided, --algorithms can only take 'cle' or 'adaround'")


def check_for_single_process_env():
    single_process = True
    log_str = None
    try:
        from packaging import version
        import torch
        import qti.aisw.converters.aimet.qnn_quantsim_adapter
        log_str = 'AIMET Quantizer will run in the same process'
        log_info(log_str)
    except ImportError or ModuleNotFoundError:
        single_process = False
        log_str = f'Using multiprocess setup to run AIMET Quantizer'
        log_info(log_str)

    return single_process


def aimet_dlc_quantizer(opts):
    is_single_process_env = check_for_single_process_env()
    aimet_env_python_exec = None
    if not is_single_process_env:
        aimet_env_python_exec = get_python_executable()
        if not aimet_env_python_exec:
            raise EnvironmentError(
                """Environment variable 'AIMET_ENV_PYTHON' not set.
                Please run  'source $QNN_SRC/QTI/scripts/aimet_env_setup.sh --env-path <PATH>' if you want to use aimet quantizer
                or omit the '--use_aimet_quantizer' flag to use the default quantizer"""
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run Torch Emitter
        backend_base_config_path = opts.torch_emitter_and_config_generator(opts.input_network, tmpdir)

        # If AIMET environment is available in current environment, run AIMET Quantizer within same process
        if is_single_process_env:
            weight_file_path, encoding_path = quantize_model_with_aimet(None, tmpdir, opts, backend_base_config_path)
        else:
            # Set multiprocessing context to 'spawn' to keep AIMET process be independent
            # of main process environment and set the python executable path
            mp.set_start_method("spawn", force=True)
            mp.set_executable(aimet_env_python_exec)
            # Create a process and run aimet-specific code within the context of that process
            parent_conn, child_conn = mp.Pipe()
            if os.environ.get("AIMET_DEBUG_DIR") is not None:
                debug_dir = os.environ.get("AIMET_DEBUG_DIR")
                if not os.path.exists(debug_dir):
                    log_info(f"Path specified in AIMET_DEBUG_DIR does not exist. Not saving debug "
                             "files.")
                else:
                    log_info(f"Saving debug files at '{debug_dir}'")
                    tmpdir = debug_dir
            process = AimetProcess(target=quantize_model_with_aimet,
                                   args=(child_conn, tmpdir, opts, backend_base_config_path),
                                   aimet_venv_python_path=aimet_env_python_exec,
                                   parent_sys_path=sys.path.copy())
            process.start()
            retval = parent_conn.recv()
            weight_file_path = retval[0]
            encoding_path = retval[1]
            process.join()
            process.terminate()

        if (weight_file_path is not None and os.path.exists(weight_file_path)) and (encoding_path is not None and
                                                                                    os.path.exists(encoding_path)):
            ##### Get Quantized DLC ######
            lwq_input_config = LightWeightQuantizerInputConfig(path=tmpdir,
                                                               filename_prefix=opts.emitter_filename,
                                                               dlc_path=opts.input_network,
                                                               weight_file_path=weight_file_path,
                                                               encoding_path=encoding_path,
                                                               quantize_dlc=True,
                                                               activation_bitwidth=opts.act_bitwidth,
                                                               float_bias_bitwidth=opts.float_bias_bw)
            lwq_quantizer = QAIRTLightWeightQuantizer()
            lwq_output_config = lwq_quantizer.export(lwq_input_config)
            quantized_dlc_path = lwq_output_config.dlc_path

            if quantized_dlc_path is not None and os.path.exists(quantized_dlc_path):
                reader = get_dlc_reader(quantized_dlc_path)
                return reader
            else:
                log_error('Exception occured in LWQ Module')
                sys.exit(1)
        else:
            log_error('Exception occured in Spawned AIMET Process, Unable to proceed with Quantization')
            sys.exit(1)