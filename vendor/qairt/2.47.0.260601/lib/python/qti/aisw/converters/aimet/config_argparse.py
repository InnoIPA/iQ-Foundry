# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================
from typing import Callable, Union
import torch
import importlib
from tqdm import tqdm
import math
import json
import itertools
import tempfile

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.amp.utils import AMPSearchAlgo
from aimet_torch import utils as aimet_utils
from aimet_common.defs import CallbackFunc
from aimet_torch.quantsim import QuantizationSimModel

import logging
logger = logging.getLogger('AIMET Quantizer')

from enum import IntEnum
class AIMET_ALGO(IntEnum):
    QUANTSIM = 1
    DEFAULT_ADAROUND = 2
    ADAROUND = 3
    AMP = 4
    AUTOQUANT = 5

# Mapping from QNN calibration to AIMET QuantScheme
QNN_TO_AIMET_QUANT_SCHEME_MAP = {
    "tf": QuantScheme.post_training_tf,
    "min-max": QuantScheme.post_training_tf,
    "enhanced": QuantScheme.post_training_tf_enhanced,
    "sqnr": QuantScheme.post_training_tf_enhanced,
    "percentile": QuantScheme.post_training_percentile,
    "symmetric": QuantScheme.post_training_tf,
}


def place_data_on_device(sample, device):
    if isinstance(sample, (tuple, list)):
        sample = tuple([tensor.to(device) for tensor in sample])
        logger.info(f'Sample input data shape, {[x.shape for x in sample]}')
    else:
        sample = sample.to(device)
        logger.info(f'Sample input data shape, {sample.shape}')
    return sample


def labeled_data_calibration_cb() -> Callable:
    def pass_calibration_data(model, args):

        dataloader, device, iterations  = args
        model.eval()

        if iterations is None:
            logger.info('No value of iteration is provided, running forward pass on complete '
                        'dataset.')
            iterations = len(dataloader)
        if iterations <= 0:
            logger.error('Cannot evaluate on %d iterations', iterations)

        batch_cntr = 1
        with torch.no_grad():
            for input_data,_ in tqdm(dataloader, desc='compute_encodings'):
                if isinstance(input_data, torch.Tensor):
                    input_batch = input_data.to(device)
                    model(input_batch)
                else:
                    input_batch = tuple(map(lambda d: d.to(device), input_data))
                    model(*input_batch)

                batch_cntr += 1
                if batch_cntr > iterations:
                    break

    return pass_calibration_data


def unlabeled_data_calibration_cb() -> Callable:
    def pass_calibration_data(model, args):

        dataloader, device, iterations = args
        model.eval()

        if iterations is None:
            logger.info('No value of iteration is provided, running forward pass on complete dataset.')
            iterations = len(dataloader)
        if iterations <= 0:
            logger.error('Cannot evaluate on %d iterations', iterations)

        batch_cntr = 1
        with torch.no_grad():
            for input_data in tqdm(dataloader, desc='compute_encodings'):
                if isinstance(input_data, torch.Tensor):
                    input_batch = input_data.to(device)
                    model(input_batch)
                else:
                    input_batch = tuple(map(lambda d: d.to(device), input_data))
                    model(*input_batch)

                batch_cntr += 1
                if batch_cntr > iterations:
                    break

    return pass_calibration_data


def input_list_data_calibration_cb() -> Callable:
    """Get the calibration callback needed for computing encodings"""
    def pass_calibration_data(model, args):

        dataloader, device, iterations = args
        model.eval()

        with torch.no_grad():
            for input_data in tqdm(dataloader, desc='compute_encodings'):
                if isinstance(input_data, torch.Tensor):
                    input_batch = input_data[0].to(device)
                    model(input_batch)
                else:
                    input_batch = tuple(map(lambda d: d[0].to(device), input_data))
                    model(*input_batch)
    return pass_calibration_data


def get_callback_function(module_string: str):
    try:
        module_name, method_name = module_string.rsplit('.', 1)
        callback_module = importlib.import_module(module_name)
        callback_function = getattr(callback_module, method_name)
    except ModuleNotFoundError as e:
        logger.error("Make sure that the callback function is located in a directory that "
                     "is part of python 'sys.path'.")
        raise RuntimeError('Error loading callback function specified in config file') from e

    except AttributeError as e:
        logger.error("Make sure that the callback function is defined in the specified path")
        raise RuntimeError('Error loading callback function specified in config file') from e

    return callback_function


def validate_data_type(args, default_dtypes):
    for param, value in args.items():
        if param in default_dtypes:
            if not isinstance(value, default_dtypes[param]):
                raise TypeError("Incorrect data type of %s in config file" % param)
        else:
            raise ValueError("Invalid argument : %s in config file" % param)


class DatasetContainer:

    def __init__(self, dataloader_callback: str, dataloader_kwargs: dict = {}):

        self.dataloader_callback = get_callback_function(dataloader_callback)
        self.dataloader_kwargs = dataloader_kwargs


class QuantSimConfig:

    def __init__(self, opts, model, dummy_input, backend_aware_base_config, input_shapes):
        self.opts = opts
        self.model = model
        if dummy_input:
            self.dummy_input = dummy_input
        else:
            self.dummy_input = self.get_sample_data(input_shapes)
        self.param_quant_scheme, self.act_quant_scheme = self._get_act_param_quant_schemes()
        self.quant_scheme = self._get_quant_scheme()
        self.rounding_mode = "nearest"
        self.default_output_bw = opts.act_bitwidth
        self.default_param_bw = opts.weights_bitwidth
        self.in_place = True
        self.config_file = self.get_qsim_config(backend_aware_base_config)
        self.default_data_type = QuantizationDataType.float if opts.float_fallback else QuantizationDataType.int
        self.forward_pass_fxn = None
        self.dataloader = None
        self.iteration_size = None

    def get_qsim_args(self):

        quantsim_args_dict = {
            "model": self.model,
            "dummy_input": self.dummy_input,
            "quant_scheme": self.quant_scheme,
            "default_output_bw": self.default_output_bw,
            "default_param_bw": self.default_param_bw,
            "in_place": self.in_place,
            "config_file": self.config_file,
            "default_data_type": self.default_data_type
        }
        return quantsim_args_dict

    def _get_act_param_quant_schemes(self):
        if self.opts.disable_legacy_quant_scheme_opts:
            # weights_bitwidth 16 is supported only with 'symmetric' on HTP
            # TODO: Sync with backend awareness
            if self.opts.weights_bitwidth == 16:
                logger.warning("Forcing param_quantizer_schema to 'symmetric', as --weights_bitwidth is 16")
                self.opts.quant_schemes['param_quant']['schema'] = 'symmetric'
            param_quant_scheme = self.opts.quant_schemes['param_quant']["calibration"]
            act_quant_scheme = self.opts.quant_schemes['act_quant']["calibration"]
        else:
            # weights_bitwidth 16 is supported only with 'symmetric' on HTP
            # TODO: Sync with backend awareness
            if self.opts.weights_bitwidth  == 16:
                logger.warning("Forcing param_quantizer to 'symmetric', as --weights_bitwidth is 16")
                self.opts.quant_schemes['param_quant'] = 'symmetric'
            param_quant_scheme = self.opts.quant_schemes['param_quant']
            act_quant_scheme = self.opts.quant_schemes['act_quant']
        return param_quant_scheme, act_quant_scheme

    def _get_quant_scheme(self) -> QuantScheme:
        """
        Get the quantization scheme from qnn arguments
        """
        # Set default Quantization scheme to param_quant_schema, and modify after QuantSim instantiation
        quant_scheme = self.param_quant_scheme
        if QNN_TO_AIMET_QUANT_SCHEME_MAP[self.param_quant_scheme] != QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme]:
            logger.info("Quantization schemes for parameter quantizers and activations quantizers are different")
        return QNN_TO_AIMET_QUANT_SCHEME_MAP.get(quant_scheme,None)

    def get_sample_data(self, input_shapes) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Get n random samples from the dataset
        :return: return random samples or  entire dataset if input list provided
        """
        dummy_input = tuple(
            torch.randn(shape)
            for shape in input_shapes.values()
        )
        if len(dummy_input) == 1:
            dummy_input = dummy_input[0]
        return dummy_input

    def set_param_for_quantsim(self, dataloader, algorithm, device, iteration_size,
                               config_dataloader = False):

        self.dataloader = dataloader
        self.iteration_size = iteration_size
        if algorithm in (AIMET_ALGO.ADAROUND, AIMET_ALGO.AUTOQUANT):
            self.forward_pass_fxn = unlabeled_data_calibration_cb()
            self.dummy_input = next(iter(self.dataloader))
            self.dummy_input = place_data_on_device(self.dummy_input, device)
        elif algorithm==AIMET_ALGO.AMP:
            self.forward_pass_fxn = labeled_data_calibration_cb()
            self.dummy_input,_ = next(iter(self.dataloader))
            self.dummy_input = place_data_on_device(self.dummy_input, device)
        else: # QuantSim and Default AdaRound
            if config_dataloader:
                self.forward_pass_fxn = unlabeled_data_calibration_cb()
                self.dummy_input = next(iter(self.dataloader))
                self.dummy_input = place_data_on_device(self.dummy_input, device)
            else:
                self.forward_pass_fxn = input_list_data_calibration_cb()
                self.dummy_input = next(iter(self.dataloader))
                if isinstance(self.dummy_input, (tuple, list)):
                    self.dummy_input = tuple([tensor[0] for tensor in self.dummy_input])
                else:
                    self.dummy_input = self.dummy_input[0]
                self.dummy_input = place_data_on_device(self.dummy_input, device)

    def _is_unsigned_symmetric_act(self):
        return self.opts.disable_legacy_quant_scheme_opts and self.opts.quant_schemes['act_quant']['schema'] == 'unsignedsymmetric'

    def _is_unsigned_symmetric_param(self):
        return self.opts.disable_legacy_quant_scheme_opts and self.opts.quant_schemes['param_quant']['schema'] == 'unsignedsymmetric'

    def _required_to_modify_act_quant(self):
        return (QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme] != self.quant_scheme
                or self._is_unsigned_symmetric_act() or self.act_quant_scheme == 'percentile')

    def _required_to_modify_param_quant(self):
        # Need not check for quant scheme, as we use param_quant_scheme for Quantsim instantiation
        return self._is_unsigned_symmetric_param() or self.param_quant_scheme == 'percentile'

    def modify_quantizers_if_needed(self, sim: QuantizationSimModel, adaround=False):

        required_to_modify_param_quant = self._required_to_modify_param_quant()
        required_to_modify_act_quant = self._required_to_modify_act_quant()

        if required_to_modify_act_quant or required_to_modify_param_quant:
            logger.info('Modifying Quantizer settings based on command line arguments..')
            param_quantizers, input_quantizers, output_quantizers = aimet_utils.get_all_quantizers(sim.model)

            if required_to_modify_act_quant:
                for quantizer in itertools.chain(input_quantizers, output_quantizers):
                    if QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme] != self.quant_scheme:
                        # Set input/output quantizers' quant schemes
                        quantizer.quant_scheme = QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme]
                    if self._is_unsigned_symmetric_act():
                        # Set the following for unsigned_symmetric
                        quantizer.use_unsigned_symmetric = True
                        quantizer.use_symmetric_encodings = True
                    if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                            self.opts.percentile_calibration_value is not None:
                        quantizer.set_percentile_value(self.opts.percentile_calibration_value)

            if required_to_modify_param_quant:
                if adaround:
                    logger.warning("Can't modify param quantizer settings now for Adaround!")
                else:
                    for quantizer in param_quantizers:
                        if self._is_unsigned_symmetric_param():
                            # Set the following for unsigned_symmetric
                            quantizer.use_unsigned_symmetric = True
                            quantizer.use_symmetric_encodings = True
                        if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                                self.opts.percentile_calibration_value is not None:
                            quantizer.set_percentile_value(self.opts.percentile_calibration_value)
        return sim

    def _force_param_quantizer_schema_to_symmetric(self):
        if self.opts.disable_legacy_quant_scheme_opts and self.opts.quant_schemes['param_quant']['schema'] == 'asymmetric':
            self.opts.quant_schemes['param_quant']['schema'] = 'symmetric'
            logger.warning("Can't use 'asymmetric' param_quantizer_schema with --use_per_channel_quantization "
                           "or --use_per_row_quantization. Using 'symmetric' param_quantizer_schema!")

    def get_qsim_config(self, backend_aware_base_config):
        # TODO: AIMET Torch is not supporting QAIRT backend configs
        # if backend_aware_base_config is None:
        from aimet_common.quantsim_config.config_utils import get_path_for_target_config
        backend_aware_base_config = get_path_for_target_config("htp_quantsim_config_v81")
        config_dict = json.load(open(backend_aware_base_config))

        force_symmetric_param_quant = False
        if self.opts.use_per_channel_quantization:
            logger.info(f'Per Channel Quantization is enabled')
            force_symmetric_param_quant = True
        if self.opts.use_per_row_quantization:
            logger.info(f'Per Row Quantizaton is enabled')
            force_symmetric_param_quant = True

        if force_symmetric_param_quant:
            self._force_param_quantizer_schema_to_symmetric()

        def add_to_config(op_type, flag):
            if op_type in config_dict["op_type"].keys():
                config_dict["op_type"][op_type]["per_channel_quantization"] = flag
            else:
                config_dict["op_type"][op_type] = {"per_channel_quantization": flag}

        # TODO: Make Pad Op a data movement Op or add additional pads within a custom module in MPP
        config_dict["op_type"]["Pad"] = {"is_output_quantized": "False"}
        config_dict["op_type"]["Buffer"] = {"is_output_quantized": "False"}

        config_dict["defaults"]["per_channel_quantization"] = str(self.opts.use_per_channel_quantization)
        # per_row_ops = ["Gemm", "MatMul"]
        add_to_config("Gemm", str(self.opts.use_per_row_quantization))
        add_to_config("MatMul", str(self.opts.use_per_row_quantization))

        quantizer_type_to_config_key = {
            'act_quant': "ops",
            'param_quant': "params"
        }

        if self.opts.disable_legacy_quant_scheme_opts:
            for quantizer in ['act_quant', 'param_quant']:
                if self.opts.disable_legacy_quant_scheme_opts:
                    if self.opts.quant_schemes[quantizer]['schema'] in ["asymmetric"]:
                        config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "False"
                    elif self.opts.quant_schemes[quantizer]['schema'] in ["symmetric"]:
                        config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "True"
                else:
                    if self.opts.quant_schemes[quantizer] == 'symmetric':
                        config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "True"

        temp_config_file = tempfile.NamedTemporaryFile(delete=False)

        with open(temp_config_file.name, "w") as file:
            json.dump(config_dict, file)

        return temp_config_file.name

    def validate_qsim_options(self):

        valid_opts = True
        if self.param_quant_scheme not in QNN_TO_AIMET_QUANT_SCHEME_MAP.keys():
            param_quantizer_arg_key = "--param_quantizer" if not self.opts.disable_legacy_quant_scheme_opts else "--param_quantizer_calibration"
            logger.error(f"invalid value '{self.param_quant_scheme}' for {param_quantizer_arg_key}")
            valid_opts = False
        if self.act_quant_scheme not in QNN_TO_AIMET_QUANT_SCHEME_MAP.keys():
            act_quantizer_arg_key = "--act_quantizer" if not self.opts.disable_legacy_quant_scheme_opts else "--act_quantizer_calibration"
            logger.error(f"invalid value '{self.act_quant_scheme}' for {act_quantizer_arg_key}")
            valid_opts = False
        return valid_opts


class AdaRoundConfig:

    def __init__(self, dataloader: str, num_batches: int, optional_adaround_param_args: dict = {},
                 optional_adaround_args: dict = {}, iteration_size: int = None):

        self.dataloader = dataloader
        self.num_batches = num_batches
        self.optional_adaround_param_args = self._validate_and_set_optional_adaround_param_args(optional_adaround_param_args)
        self.optional_adaround_args = self._validate_and_set_optional_adaround_args(optional_adaround_args)
        self.result_dir = None
        self.forward_pass_fxn = unlabeled_data_calibration_cb()
        self.iteration_size = iteration_size

    def set_param_for_default_adaround(self, input_list_dataloader):
        self.dataloader = input_list_dataloader
        self.num_batches = num_batches = min(len(self.dataloader), math.ceil(2000/self.dataloader.batch_size))
        self.forward_pass_fxn = input_list_data_calibration_cb()

    def _validate_and_set_optional_adaround_param_args(self, optional_adaround_param_args):

        default_adaround_param_args_dtype = {'default_num_iterations': int,
                                             'default_reg_param': float,
                                             'default_beta_range': list,
                                             'default_warm_start': float,
                                             'forward_fn': (Callable, str)
                                             }
        validate_data_type(optional_adaround_param_args, default_adaround_param_args_dtype)

        if 'default_beta_range' in optional_adaround_param_args:
            optional_adaround_param_args['default_beta_range'] = tuple(optional_adaround_param_args['default_beta_range'])

        if ('forward_fn' in optional_adaround_param_args and
                isinstance(optional_adaround_param_args['forward_fn'], str)):
            optional_adaround_param_args['forward_fn'] = get_callback_function(optional_adaround_param_args['forward_fn'])

        return optional_adaround_param_args

    def _validate_and_set_optional_adaround_args(self,optional_adaround_args):

        default_adaround_args_dtype = {'default_param_bw': int,
                                       'param_bw_override_list': list,
                                       'ignore_quant_ops_list': list,
                                       'default_quant_scheme': str,
                                       'default_config_file': str
                                       }

        validate_data_type(optional_adaround_args, default_adaround_args_dtype)

        if 'default_quant_scheme' in optional_adaround_args:
            optional_adaround_args['default_quant_scheme'] = QuantScheme[optional_adaround_args['default_quant_scheme']]

        return optional_adaround_args


class AMPConfig:

    def __init__(self, dataloader: torch.utils.data.dataloader.DataLoader, candidates: list, allowed_accuracy_drop: float, eval_callback_for_phase2: str,
                 optional_amp_args: dict = {}, iteration_size: int = None):

        self.dataloader = dataloader
        self.eval_callback_for_phase2 = eval_callback_for_phase2
        self.candidates = self._validate_and_set_amp_candidates(candidates)
        self.allowed_accuracy_drop = allowed_accuracy_drop
        self.forward_pass_fxn = labeled_data_calibration_cb()
        self.result_dir = None
        self.iteration_size = iteration_size

        if not isinstance(allowed_accuracy_drop, (type(None), float)):
            raise TypeError("allowed_accuracy_drop in config file")

        self.optional_amp_args = self._validate_and_set_optional_amp_args(optional_amp_args)

    def _validate_and_set_amp_candidates(self, candidates):

        candidate_list = []
        for candidate in candidates:
            [[output_bw, output_dtype], [param_bw, param_dtype]] = candidate
            updated_candidate = ((output_bw, QuantizationDataType[output_dtype]), (param_bw, QuantizationDataType[param_dtype]))
            candidate_list.append(updated_candidate)

        #TODO Validation on candidates allowed on target
        return candidate_list

    def _validate_and_set_optional_amp_args(self,optional_amp_args):

        default_amp_args_dtype = {
            'eval_callback_for_phase1': (CallbackFunc, str),
            'forward_pass_callback': (CallbackFunc, str),
            'use_all_amp_candidates': bool,
            'phase2_reverse': bool,
            'amp_search_algo': str,
            'clean_start': bool
        }

        validate_data_type(optional_amp_args, default_amp_args_dtype)

        if ('eval_callback_for_phase1' in optional_amp_args and
                isinstance(optional_amp_args['eval_callback_for_phase1'], str)):
            optional_amp_args['eval_callback_for_phase1'] = CallbackFunc(get_callback_function(
                optional_amp_args['eval_callback_for_phase1']), None)

        if ('forward_pass_callback' in optional_amp_args and
                isinstance(optional_amp_args['forward_pass_callback'], str)):
            optional_amp_args['forward_pass_callback'] = CallbackFunc(get_callback_function(
                optional_amp_args['forward_pass_callback']), None)

        if 'amp_search_algo' in optional_amp_args:
            optional_amp_args['amp_search_algo'] = AMPSearchAlgo[optional_amp_args['amp_search_algo']]

        return optional_amp_args

class AutoQuantConfig:

    def __init__(self, dataloader: torch.utils.data.dataloader.DataLoader,
                 eval_callback: str, eval_dataloader: str, allowed_accuracy_drop: float,
                 amp_candidates: list = None, optional_autoquant_args: dict = {},
                 optional_adaround_args: dict = {}, optional_amp_args: dict = {},
                 iteration_size: int = None):

        self.dataloader = dataloader
        self.eval_callback = eval_callback
        self.eval_dataloader = eval_dataloader
        self.allowed_accuracy_drop = allowed_accuracy_drop
        self.amp_candidates = self._validate_and_set_amp_candidates(amp_candidates)
        self.forward_pass_fxn = unlabeled_data_calibration_cb()
        self.result_dir = None
        self.iteration_size = iteration_size

        if not isinstance(allowed_accuracy_drop, float):
            raise TypeError("allowed_accuracy_drop in config file")

        self.optional_autoquant_args = self._validate_and_set_optional_autoquant_args(optional_autoquant_args)
        self.optional_adaround_args = self._validate_and_set_optional_adaround_args(optional_adaround_args)
        self.optional_amp_args = self._validate_and_set_optional_amp_args(optional_amp_args)


    def _validate_and_set_amp_candidates(self, candidates):

        candidate_list = []
        for candidate in candidates:
            [[output_bw, output_dtype], [param_bw, param_dtype]] = candidate
            updated_candidate = ((output_bw, QuantizationDataType[output_dtype]), (param_bw, QuantizationDataType[param_dtype]))
            candidate_list.append(updated_candidate)

        #TODO Validation on candidates allowed on target
        return candidate_list

    def _validate_and_set_optional_autoquant_args(self, optional_autoquant_args):

        default_autoquant_args_dtype = {
            'param_bw': int,
            'output_bw': int,
            'quant_scheme': str,
            'rounding_mode': str,
            'config_file': str,
            'cache_id': str,
            'strict_validation': bool
        }

        validate_data_type(optional_autoquant_args, default_autoquant_args_dtype)

        if 'quant_scheme' in optional_autoquant_args:
            optional_autoquant_args['quant_scheme'] = QuantScheme[optional_autoquant_args['quant_scheme']]

        return optional_autoquant_args

    def _validate_and_set_optional_adaround_args(self, optional_adaround_args):

        default_adaround_args_dtype = {
            'num_batches': int,
            'default_num_iterations': int,
            'default_reg_param': float,
            'default_beta_range': list,
            'default_warm_start': float,
            'forward_fn': (Callable, str)
        }

        validate_data_type(optional_adaround_args, default_adaround_args_dtype)

        if 'default_beta_range' in optional_adaround_args:
            optional_adaround_args['default_beta_range'] = tuple(optional_adaround_args['default_beta_range'])

        if ('forward_fn' in optional_adaround_args and
                isinstance(optional_adaround_args['forward_fn'], str)):
            optional_adaround_args['forward_fn'] = get_callback_function(optional_adaround_args['forward_fn'])

        return optional_adaround_args

    def _validate_and_set_optional_amp_args(self, optional_amp_args):

        default_amp_args_dtype = {
            'num_samples_for_phase_1': int,
            'forward_fn': (Callable, str),
            'num_samples_for_phase_2': int
        }

        validate_data_type(optional_amp_args, default_amp_args_dtype)

        if 'forward_fn' in optional_amp_args and isinstance(optional_amp_args['forward_fn'], str):
            optional_amp_args['forward_fn'] = get_callback_function(optional_amp_args['forward_fn'])

        return optional_amp_args

def select_algorithm(algo_list, quantizer_algorithm):
    '''
    Selects the algorithm to be run.
    :param algo_list: Aimet algorithms present in config file.
    :param quantizer_algorithm: Algorithms provided by User in Quantizer option.
    :return:
    '''

    if len(algo_list)==1:
        return algo_list[0]
    elif len(algo_list)>1:
        logger.info("Multiple config provided")
        selected_algo = [algo for algo in algo_list if algo in quantizer_algorithm]
        if len(selected_algo)==0:
            raise ValueError("Invalid algorithm provided for specified config file")
        else:
            logger.info(f"Using '{selected_algo[0]}' for quantization.")
            return selected_algo[0]
    else:
        return 'quantsim'

def set_dataloader(algo_config: dict, dataset_dict: dict, dataset_arg = 'dataset',
                   dataloader_arg = 'dataloader'):

    if algo_config[dataset_arg] not in dataset_dict.keys():
        raise Exception("dataset configuration not defined in YAML file")

    dataset_name = algo_config.pop(dataset_arg)
    dataset_container = DatasetContainer(**dataset_dict[dataset_name])
    dataloader = dataset_container.dataloader_callback(**dataset_container.dataloader_kwargs)
    algo_config[dataloader_arg] = dataloader

    return algo_config

def get_algorithm_config(config, quantizer_algorithm):

    ## CMDLine Flow Conditions, in which has to pass config yaml file for AdaRound, AMP, AutoQuant and pass
    ## 'adaround' in algorithm for default-AdaRound and flag --use_aimet_quantizer for QuantSim
    advance_quantizer_list =  [key.lower() for key, _ in AIMET_ALGO.__members__.items()]
    ## No Config Defined
    if config is None:
        # QuantSim
        if not set(quantizer_algorithm) & set(advance_quantizer_list):
            return None, AIMET_ALGO.QUANTSIM
        # default_AdaRound
        if 'adaround' in quantizer_algorithm:
            return AdaRoundConfig(None, None), AIMET_ALGO.DEFAULT_ADAROUND
        else:
            raise ValueError(f"Invalid algorithm provided!")

    else:
        quantizers_in_config = [k for k in advance_quantizer_list if k in config.keys()]
        if len(quantizers_in_config)<1:
            raise ValueError(f'Invalid config provided')

        aimet_algorithm = select_algorithm(quantizers_in_config, quantizer_algorithm)

        if aimet_algorithm == 'adaround':
            if config['adaround'].get('num_batches', None) is None:
                return AdaRoundConfig(None, None), AIMET_ALGO.DEFAULT_ADAROUND
            else:
                if 'datasets' in config.keys():
                    config['adaround'] = set_dataloader(config['adaround'], config['datasets'])
                    config['adaround']['iteration_size'] = config['adaround'].get('iteration_size', None)
                return AdaRoundConfig(**config['adaround']), AIMET_ALGO.ADAROUND

        if aimet_algorithm=="amp":
            ## CMDLine Flow
            if 'datasets' in config.keys():
                config['amp'] = set_dataloader(config['amp'], config['datasets'])
                config['amp']['eval_callback_for_phase2'] = get_callback_function(config['amp']['eval_callback_for_phase2'])
                config['amp']['iteration_size'] = config['amp'].get('iteration_size', None)
            return AMPConfig(**config['amp']), AIMET_ALGO.AMP

        if aimet_algorithm=="autoquant":
            ## CMDLine Flow
            if 'datasets' in config.keys():
                config['autoquant'] = set_dataloader(config['autoquant'], config['datasets'])
                config['autoquant'] = set_dataloader(config['autoquant'], config['datasets'], dataset_arg = 'eval_dataset',
                                                     dataloader_arg = 'eval_dataloader')
                config['autoquant']['eval_callback'] = get_callback_function(config['autoquant']['eval_callback'])
                config['autoquant']['iteration_size'] = config['autoquant'].get('iteration_size', None)
            return AutoQuantConfig(**config['autoquant']), AIMET_ALGO.AUTOQUANT

        else:
            if 'datasets' in config.keys():
                config['quantsim'] = set_dataloader(config['quantsim'], config['datasets'])
                config['quantsim']['iteration_size'] = config['quantsim'].get('iteration_size', None)
            return config.get('quantsim', None), AIMET_ALGO.QUANTSIM
