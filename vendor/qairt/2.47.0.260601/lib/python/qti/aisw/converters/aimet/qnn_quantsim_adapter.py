# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
import time
import sys
import traceback
import warnings
import re
from tqdm import tqdm
from typing import Union

logger = logging.getLogger('AIMET Quantizer')
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    from aimet_common.defs import QuantizationDataType, QuantScheme
    from aimet_torch import utils
    from aimet_torch.quantsim_config import quantsim_config
    from aimet_torch.cross_layer_equalization import equalize_model
    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
    from aimet_torch.amp.mixed_precision_algo import EvalCallbackFactory
    from aimet_torch.mixed_precision import choose_mixed_precision
    from aimet_common.defs import CallbackFunc
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.auto_quant import AutoQuantWithAutoMixedPrecision as AutoQuant

    import qti.aisw.converters.aimet.utils as aimet_utils
    from qti.aisw.converters.aimet.config_argparse import (get_algorithm_config, QuantSimConfig,
                                                           AdaRoundConfig, AMPConfig, AutoQuantConfig, AIMET_ALGO)
    from qti.aisw.emitter.utils.torch_utils import load_torch_model_using_safetensors
    from qti.aisw.converters.aimet.stateful_op_utils import (_apply_stateful_op_qsim_exception_rules,
                                                             _reset_qsim_stateful_op_buffers)
    from qti.aisw.emitter.utils.model_preparer_utils import PreparedModelInfo

except ModuleNotFoundError as e:
    traceback.print_exc()
    logger.error("Unable to import required modules to run AIMET Quantizer.")
    sys.exit(1)


class InputListDataset(Dataset):

    def __init__(self, input_list_path: str, input_shapes: dict, input_dtypes: dict, use_native_input_files: bool) -> None:
        super().__init__()
        self._input_list = open(input_list_path).readlines()
        self._input_list = [x for x in self._input_list if x != '\n']
        self._input_shapes = input_shapes
        self._is_input_list_formatted = True
        self._use_native_input_files = use_native_input_files
        # Creates a list of ordered input file list for each input based on input ordering in Ir graph
        # This ordering is inferred from the input_shapes dict as dictionaries are ordered in python>=3.7
        self._ordered_input_list = list(map(self._order_input_files, self._input_list))
        self._input_dtypes = input_dtypes

    def __len__(self):
        return len(self._input_list)

    @staticmethod
    def sanitize_name(name: str):
        name = re.sub(r'[^a-zA-Z0-9]', "_", name)
        if not (name[0].isalpha() or name[0] == "_"):
            name = "_" + name
        return name

    def _read_raw_data(self, file_name: str, dtype: str) -> np.ndarray:
        """ Read data from the .raw files into a numpy array """
        with open(file_name, "rb") as file:
            raw_data = file.read()
            if self._use_native_input_files:
                numpy_from_raw_data = np.frombuffer(raw_data, dtype=dtype)
            else:
                numpy_from_raw_data = np.frombuffer(raw_data, dtype=np.float32)
                numpy_from_raw_data = numpy_from_raw_data.astype(dtype, copy=False)
            return numpy_from_raw_data

    def _order_input_files(self, input_files):
        """ Order input files based on IR graph input name(s) """
        input_files = input_files.split() # Inputs separated by space
        is_formatted = [':=' in x for x in input_files]
        assert all(is_formatted) or not any(is_formatted), ("Input list is not well formatted")
        if all(is_formatted):
            input_files_dict = {y[0]:y[1] for y in [x.split(':=') for x in input_files]}
            try:
                input_files = [input_files_dict[input_name] for input_name in self._input_shapes.keys()]
            except KeyError:
                try:
                    input_files = [input_files_dict[self.sanitize_name(input_name)] for input_name in self._input_shapes.keys()]
                except KeyError:
                    raise KeyError("Provided incorrect input names in the input_list!!")
        else:
            # Print warning message only once
            if len(input_files) > 1 and self._is_input_list_formatted:
                self._is_input_list_formatted = False
                logger.warning("Input list is not properly formatted, may result in errors. "
                               "Write input list with input_name appended before file path "
                               "for each input, input_name:=<filepath> ..")
        return input_files

    def __getitem__(self, index) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Get the input tensor(s) at a given index of the input_list file
        Input files can be specified with the below format for three sets of inputs for two input layers

        Input_1:=Placeholder_1/real_input_inputs_1/0-0#67c965.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/1-1#54f1ff.rawtensor
        Input_1:=Placeholder_1/real_input_inputs_1/1-0#b42dc6.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/2-1#346a0e.rawtensor
        Input_1:=Placeholder_1/real_input_inputs_1/2-0#e6fb51.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/0-1#8a171b.rawtensor
        """
        ordered_input_files = self._ordered_input_list[index]

        tensors: list[torch.Tensor] = []
        for n, file_name in enumerate(ordered_input_files):
            tensor_name, tensor_dim = list(self._input_shapes.items())[n]
            tensor_dtype=self._input_dtypes[tensor_name]
            raw_data_numpy_array = self._read_raw_data(file_name, dtype=tensor_dtype)
            assert raw_data_numpy_array.shape[0] == np.prod(tensor_dim), (f'Could not reshape input tensor "{tensor_name}" '
                                                                          f'as required. Raw Numpy Data Shape: {raw_data_numpy_array.shape}; '
                                                                          f'Required tensor shape: {tensor_dim}')
            reshaped_numpy_array = raw_data_numpy_array.reshape(tensor_dim)
            tensor = torch.tensor(reshaped_numpy_array)
            tensors.append(tensor)

        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)


class QnnToAimetAdapter:

    def __init__(self, opts, backend_base_config_path, datadir: str = "",
                 use_cuda: bool = True):

        self.opts = opts
        self.aimet_config, self.aimet_algorithm = get_algorithm_config(
            self.opts.config, self.opts.algorithms)
        self._device = self._set_device(use_cuda)
        self._valid_opts = True

        if not self.opts.input_shapes:
            logger.error("Could not infer model input shapes from the model. Please specify --input_dim")
            self._valid_opts = False
        self._validate_advance_aimet_opts()
        if not self.is_valid_opts():
            raise ValueError("Invalid argument provided. Advance Quantization option can't be used")

        ### File I/O ####
        self.filename = opts.emitter_filename
        self._model_name = opts.emitter_model_name
        self.datadir = datadir
        if self.opts.output_path is not None:
            self.output_name, _ = os.path.splitext(os.path.basename(self.opts.output_path))
            self.output_dir = os.path.dirname(os.path.realpath(self.opts.output_path))
        else:
            self.output_name = self.filename
            self.output_dir = os.path.dirname(os.path.realpath(self.opts.input_network))
        if self.aimet_config and not isinstance(self.aimet_config, dict):
            results_dir = os.path.join(self.output_dir, f'{self.aimet_algorithm.name.lower()}_results_{time.strftime("%d%b%Y_%H-%M-%S")}')
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            self.aimet_config.result_dir = results_dir

        ### Emitter Model ###
        self._prepared_model = load_torch_model_using_safetensors(self._model_name, self.datadir,
                                                                  self.filename)
        self.prep_model_info = PreparedModelInfo.load_from_metadata(self._prepared_model.mpp_meta)
        self._prepared_model.to(self._device)
        self.enable_backend_aware = backend_base_config_path is not None

        ### QuantSim ###
        self.quantsim_config = QuantSimConfig(opts, self._prepared_model, None, backend_base_config_path, self.opts.input_shapes)
        self.quantsim_config.validate_qsim_options()

        ### Default DataLoader ####
        self.input_list_dataloader = None
        self.set_qsim_params()

        if self.should_run_cle():
            input_shapes = [tuple(x) for x in self.opts.input_shapes.values()]
            if len(input_shapes) == 1:
                input_shapes = input_shapes[0]
            equalize_model(self._prepared_model, input_shapes)

    def set_qsim_params(self):

        if self.aimet_algorithm == AIMET_ALGO.QUANTSIM:
            if (self.aimet_config is not None and self.aimet_config.get('dataloader', None) is not
                    None):
                self.quantsim_config.set_param_for_quantsim(self.aimet_config['dataloader'],
                                                            self.aimet_algorithm, self._device,
                                                            self.aimet_config['iteration_size'],
                                                            True)
            else:
                if self.opts.input_list:
                    self.input_list_dataloader = self.get_input_list_dataloader()
                    self.quantsim_config.set_param_for_quantsim(self.input_list_dataloader,
                                                                self.aimet_algorithm, self._device,
                                                                None, False)
        elif self.aimet_algorithm == AIMET_ALGO.DEFAULT_ADAROUND:
            if self.opts.input_list:
                self.input_list_dataloader = self.get_input_list_dataloader()
                self.aimet_config.set_param_for_default_adaround(self.input_list_dataloader)
                self.quantsim_config.set_param_for_quantsim(self.input_list_dataloader,
                                                            self.aimet_algorithm, self._device,
                                                            None,
                                                            False)

        else:
            #In case of AdaRound, AMP and AutoQuant, the dataloader used in algorithm will be
            # used for QuantSim compute_encodings.
            self.quantsim_config.set_param_for_quantsim(self.aimet_config.dataloader,
                                                        self.aimet_algorithm, self._device,
                                                        self.aimet_config.iteration_size, True)

    def get_input_list_dataloader(self):
        input_list = aimet_utils.get_input_list_abs(self.opts.input_list, self.output_name, self.datadir)
        logger.info(f"Input data type info: {self.opts.input_dtypes}")
        input_list_dataset = InputListDataset(input_list, self.opts.input_shapes, self.opts.input_dtypes, self.opts.use_native_input_files)
        # While iterating through the dataloader, the tensors have an additional pseudo dimensions (added by Torch DataLoader class).
        # So, while using it, we squeeze the additional pseudo-dimension. For example, if the input_tensor has dimensions [1, 224, 224, 3],
        # while iterating through the dataloader, it will be [1, 1, 224, 224, 3]. So, we squeeze the pseudo-dimension by using input_tensor[0].
        input_list_dataloader = DataLoader(input_list_dataset, batch_size=1, shuffle=False)
        return input_list_dataloader

    def _set_device(self, use_cuda):
        """ Set device for Quantsim """
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if use_cuda and not torch.cuda.is_available():
            logger.warning("Using 'cpu' for Quantization by AIMET Quantizer, as torch is not compiled with CUDA")
        else:
            logger.info(f"Using '{device}' for Quantization by AIMET Quantizer")
        return device

    def is_valid_opts(self):
        """ Returns whether aimet quantizer can be used with the given arguments """
        return self._valid_opts

    def _validate_advance_aimet_opts(self):
        """ Validate the command-line opts to check if aimet quantizer can be used """

        if self.aimet_algorithm in (AIMET_ALGO.AMP, AIMET_ALGO.AUTOQUANT): #AMP or AutoQuant
            if self.opts.input_list:
                logger.warning(f'--input_list is not required for {self.opts.algorithms}. '
                               f'Using dataloader, provided as callback, through config')
        elif self.aimet_algorithm == AIMET_ALGO.DEFAULT_ADAROUND: # Default-Adaround
            if not self.opts.input_list:
                logger.error("Either '--input_list' or '--config' must be specified for Adaround!")
                self._valid_opts = False
        elif self.aimet_algorithm in (AIMET_ALGO.DEFAULT_ADAROUND, AIMET_ALGO.ADAROUND):
            if self.opts.disable_legacy_quant_scheme_opts and self.opts.quant_schemes['param_quant']['schema'] == 'unsignedsymmetric':
                logger.error(f"Unsigned symmetric quantizer schema is not supported for params during Adaround!")
                self._valid_opts = False
        else:
            if not self.opts.input_list and not self.opts.float_fallback:
                logger.error("'--input_list' or '--float_fallback' needs to be specified")
                self._valid_opts = False


    def should_run_cle(self) -> bool:
        """
        Returns true if cle should be run on the model before quantization
        """
        return "cle" in self.opts.algorithms

    def should_run_adaround(self) -> bool:
        """
        Returns true if adaround should be run on the model
        """
        return self.aimet_algorithm in (AIMET_ALGO.DEFAULT_ADAROUND, AIMET_ALGO.ADAROUND)

    def should_run_amp(self) -> bool:
        """
        Returns true if amp should be run on the sim model
        """
        return self.aimet_algorithm == AIMET_ALGO.AMP

    def should_run_autoquant(self) -> bool:
        """
        Returns true if autoquant should be run on the sim model
        """
        return self.aimet_algorithm == AIMET_ALGO.AUTOQUANT

    def get_prepare_model(self):
        return self._prepared_model

    def _set_adaround_param(self):
        """
        Set Adaround parameters.
        """
        def _default_forward_fn(model, inputs):
            '''
            Used in case of input_list dataloader
            '''
            inputs = inputs[0]
            if isinstance(inputs, torch.Tensor):
                input_batch = inputs.to(self._device)
                return model(input_batch)
            assert isinstance(inputs, (tuple, list))
            input_batch = tuple(map(lambda d: d.to(self._device), inputs))
            return model(*input_batch)

        if self.aimet_algorithm == AIMET_ALGO.DEFAULT_ADAROUND:
            _adaround_params = AdaroundParameters(data_loader=self.aimet_config.dataloader,
                                                  num_batches=self.aimet_config.num_batches,
                                                  forward_fn=_default_forward_fn,
                                                  default_num_iterations = 1)
        else:
            _adaround_params = AdaroundParameters(data_loader=self.aimet_config.dataloader,
                                                  num_batches=self.aimet_config.num_batches,
                                                  **self.aimet_config.optional_adaround_param_args)

        return _adaround_params

    def _apply_adaround(self):
        """
        Calls AdaRound API in AIMET and saves the param encoding in a file.
        """

        adaround_params = self._set_adaround_param()

        #Config file values will overide the default values of QAIRT/AIMET
        #In case the values are not provided in the config file, default/user-provided QAIRT values will be applied.
        if 'default_param_bw' not in self.aimet_config.optional_adaround_args:
            self.aimet_config.optional_adaround_args['default_param_bw'] = self.quantsim_config.default_param_bw
        else:
            self.quantsim_config.default_param_bw = self.aimet_config.optional_adaround_args['default_param_bw']

        if 'default_quant_scheme' not in self.aimet_config.optional_adaround_args:
            logger.info(f"Using '{self.quantsim_config.quant_scheme}' from command line "
                        f"as param quant scheme, as it is not mentioned in config")
            self.aimet_config.optional_adaround_args['default_quant_scheme'] = self.quantsim_config.quant_scheme
        else:
            logger.info(f"Using '{self.aimet_config.optional_adaround_args['default_quant_scheme']}'"
                        f" from adaround config as param quant scheme")
            self.quantsim_config.quant_scheme = self.aimet_config.optional_adaround_args['default_quant_scheme']

        if 'default_config_file' not in self.aimet_config.optional_adaround_args:
            self.aimet_config.optional_adaround_args['default_config_file'] = self.quantsim_config.config_file
        else:
            self.quantsim_config.config_file = self.aimet_config.optional_adaround_args['default_config_file']

        ada_model = Adaround.apply_adaround(self._prepared_model, self.quantsim_config.dummy_input, adaround_params,
                                            path=self.aimet_config.result_dir,
                                            filename_prefix="adaround",
                                            **self.aimet_config.optional_adaround_args
                                            )
        adaround_encoding_path = os.path.join(self.aimet_config.result_dir,
                                              "{}.encodings".format("adaround"))
        self.quantsim_config.model = ada_model
        return adaround_encoding_path

    def _set_amp_param(self):

        def forward_one_batch(model, batch):
            model.to(self._device)
            model.eval()
            sample, label = batch
            if isinstance(sample, torch.Tensor):
                sample = sample.to(self._device)
                return model(sample)
            else:
                sample = tuple(map(lambda d: d.to(self._device), sample))
                return model(*sample)

        def evaluator(model, eval_function):

            result = 0.0
            model.eval()
            with torch.no_grad():
                for input_data, target_data in tqdm(self.aimet_config.dataloader, desc='eval'):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data.to(self._device)
                        predicted_batch = model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d.to(self._device), input_data))
                        predicted_batch = model(*input_batch)
                    target_data = target_data.to(self._device) if isinstance(target_data, torch.Tensor) else tuple(map(lambda d: d.to(self._device), target_data))
                    out = eval_function(output=predicted_batch, target=target_data)
                    result += out[0].item()
            result /= len(self.aimet_config.dataloader)
            return result

        amp_args = {}
        amp_args['candidates'] = self.aimet_config.candidates

        if 'eval_callback_for_phase1' in self.aimet_config.optional_amp_args:
            amp_args['eval_callback_for_phase1'] = self.aimet_config.optional_amp_args['eval_callback_for_phase1']
        else:
            factory = EvalCallbackFactory(self.aimet_config.dataloader, forward_fn=forward_one_batch)
            amp_args['eval_callback_for_phase1'] = factory.sqnr(EvalCallbackFactory._DEFAULT_SQNR_NUM_SAMPLES)
        amp_args['eval_callback_for_phase2'] = CallbackFunc(evaluator, self.aimet_config.eval_callback_for_phase2,)
        amp_args['allowed_accuracy_drop'] = self.aimet_config.allowed_accuracy_drop

        amp_args['results_dir'] = self.aimet_config.result_dir

        amp_args['clean_start'] = self.aimet_config.optional_amp_args.get("clean_start", True)
        amp_args['forward_pass_callback'] = self.aimet_config.optional_amp_args.get(
            'forward_pass_callback',CallbackFunc(self.aimet_config.forward_pass_fxn,
                                                 (self.aimet_config.dataloader, self._device,
                                                  self.quantsim_config.iteration_size)))

        default_args = ['use_all_amp_candidates', 'phase2_reverse', 'amp_search_algo']
        for args in default_args:
            if args in self.aimet_config.optional_amp_args:
                amp_args[args] = self.aimet_config.optional_amp_args[args]

        return amp_args

    def _apply_amp(self, sim):

        amp_args = self._set_amp_param()
        sim.compute_encodings(self.aimet_config.forward_pass_fxn,
                              (self.quantsim_config.dataloader, self._device, self.quantsim_config.iteration_size))
        choose_mixed_precision(sim, self.quantsim_config.dummy_input, **amp_args)

    def _set_autoquant_param(self, evaluator):

        optional_autoquant_args = self.aimet_config.optional_autoquant_args
        autoquant_to_qsim_args = {
            'param_bw': 'default_param_bw',
            'output_bw': 'default_output_bw',
            'quant_scheme': 'quant_scheme',
            'rounding_mode': 'rounding_mode',
            'config_file': 'config_file'
        }
        for arg in autoquant_to_qsim_args:
            if arg in self.aimet_config.optional_autoquant_args:
                setattr(self.quantsim_config, autoquant_to_qsim_args[arg], self.aimet_config.optional_autoquant_args[arg])
            else:
                self.aimet_config.optional_autoquant_args[arg] = getattr(self.quantsim_config, autoquant_to_qsim_args[arg])

        auto_quant = AutoQuant(self._prepared_model,
                               dummy_input=self.quantsim_config.dummy_input,
                               data_loader=self.aimet_config.dataloader,
                               eval_callback=evaluator,
                               results_dir=self.aimet_config.result_dir,
                               model_prepare_required=False,
                               **self.aimet_config.optional_autoquant_args)

        # TODO: Support QuantschemePairs and Default QuantschemePairs Candidates through config file

        def _default_forward_fn(model, inputs):
            if isinstance(inputs, torch.Tensor):
                input_batch = inputs.to(self._device)
                return model(input_batch)
            assert isinstance(inputs, (tuple, list))
            input_batch = tuple(map(lambda d: d.to(self._device), inputs))
            return model(*input_batch)

        # Set AdaRound Params
        if 'num_batches' in self.aimet_config.optional_adaround_args:
            logger.info('Setting AdaRound Params in AutoQuant')
            if 'forward_fn' not in self.aimet_config.optional_adaround_args:
                self.aimet_config.optional_adaround_args['forward_fn'] = _default_forward_fn

            adaround_params = AdaroundParameters(self.aimet_config.dataloader,
                                                 **self.aimet_config.optional_adaround_args)
            auto_quant.set_adaround_params(adaround_params)

        # AMP will be enabled when two or more candidates are provided
        if (self.aimet_config.amp_candidates is not None and
                len(self.aimet_config.amp_candidates) >= 2):
            # In case the QuantSim param bw and act bw not in candidates.
            baseline_param_bw = auto_quant._auto_quant_base._quantsim_params["param_bw"]
            baseline_output_bw = auto_quant._auto_quant_base._quantsim_params["output_bw"]
            baseline_candidate = (
                (baseline_output_bw, QuantizationDataType.int),
                (baseline_param_bw, QuantizationDataType.int),
            )
            if baseline_candidate not in self.aimet_config.amp_candidates:
                self.aimet_config.amp_candidates.append(baseline_candidate)
            # Set AMP Params
            auto_quant.set_mixed_precision_params(
                candidates=self.aimet_config.amp_candidates,
                **self.aimet_config.optional_amp_args
            )

        return auto_quant

    def _apply_autoquant(self):

        def evaluator(model: torch.nn.Module, *args):

            result = 0.0
            model.to(self._device)
            model.eval()
            eval_function = self.aimet_config.eval_callback
            # labelled dataloader for evaluation
            with torch.no_grad():
                for input_data, target_data in tqdm(self.aimet_config.eval_dataloader, desc='eval'):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data.to(self._device)
                        predicted_batch = model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d.to(self._device), input_data))
                        predicted_batch = model(*input_batch)
                    target_data = target_data.to(self._device) if isinstance(target_data, torch.Tensor) else tuple(map(lambda d: d.to(self._device), target_data))
                    out = eval_function(output=predicted_batch, target=target_data)
                    result += out[0].item()
            result /= len(self.aimet_config.eval_dataloader)
            return result

        auto_quant = self._set_autoquant_param(evaluator)
        model, optimized_accuracy, encoding_path, pareto_front = auto_quant.optimize(
            allowed_accuracy_drop=self.aimet_config.allowed_accuracy_drop
        )
        return model, encoding_path

    def get_qsim(self):

        qsim_args = self.quantsim_config.get_qsim_args()
        sim = QuantizationSimModel(**qsim_args)
        return sim

    def _check_and_apply_stateful_op_exception_rules(self, sim):
        """
        If the prepared model has stateful Op, apply Qsim exception
        rules and reset buffers to zero, before computing encodings.
        """
        if self.prep_model_info.has_stateful_op:
            _apply_stateful_op_qsim_exception_rules(sim)
            _reset_qsim_stateful_op_buffers(sim)

    def _raise_error_for_stateful_op(self):
        """
        Error handle for stateful Op support
        """
        if self.prep_model_info.has_stateful_op:
            raise RuntimeError("AIMET Quantizer only supports normal Quantsim calibration for Stateful Ops")

    def generate_weight_and_encoding(self):
        """
        Returns the path of the weight and encoding of the converted, quantized IR graph
        The DLC reader needs to persist to prevent garbage collection of IRGraph static tensor data
        """

        # Check for any quantization overrides
        mpp_aligned_torch_enc_file = os.path.join(self.datadir, self.filename+'_torch.encoding')
        quantization_overrides = os.path.exists(mpp_aligned_torch_enc_file)

        if self.enable_backend_aware:
            quantsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = True

        if self.opts.float_fallback:
            sim = self.get_qsim()
            if quantization_overrides:
                self._raise_error_for_stateful_op()
                sim.load_encodings(mpp_aligned_torch_enc_file, strict=False, partial=True, allow_overwrite=False)
            else:
                raise RuntimeError('--float_fallback can only be provided when user specify encodings through --quantization_overrides')

        elif self.should_run_adaround():
            #AdaRound Flow
            adaround_encoding_path = self._apply_adaround()
            # Initialize Quantsim
            sim = self.get_qsim()
            # For adaround flow, modify quantizer setting only for activations if required based on cmd line args
            sim = self.quantsim_config.modify_quantizers_if_needed(sim, True)
            self._raise_error_for_stateful_op()
            logger.info(f'Using "{self.quantsim_config.act_quant_scheme}" as activation quant calibration scheme')
            if quantization_overrides:
                error_str = f"AdaRound cannot be used with Quantization Overrides!"
                logger.error(error_str)
                raise RuntimeError(error_str)
            else:
                # Set and freeze param encoding in case of AdaRound
                sim.set_and_freeze_param_encodings(adaround_encoding_path)
                sim.compute_encodings(self.quantsim_config.forward_pass_fxn,
                                      (self.quantsim_config.dataloader, self._device,
                                       self.quantsim_config.iteration_size))

        elif self.should_run_amp():
            if quantization_overrides:
                error_str = f"AMP cannot be used with Quantization Overrides!"
                logger.error(error_str)
                raise RuntimeError(error_str)
            else:
                logger.info('Running AMP with given config....')
                sim = self.get_qsim()
                # Modify quantizer setting for activations and params if required based on cmd line args
                sim = self.quantsim_config.modify_quantizers_if_needed(sim)
                self._raise_error_for_stateful_op()
                logger.info(f'Using "{self.quantsim_config.act_quant_scheme}" as activation quant calibration scheme '
                            f'and "{self.quantsim_config.param_quant_scheme}" as param quant calibration scheme')
                self._apply_amp(sim)

        elif self.should_run_autoquant():
            if quantization_overrides:
                error_str = f"AutoQuant cannot be used with Quantization Overrides!"
                logger.error(error_str)
                raise RuntimeError(error_str)
            else:
                logger.info('Running AutoQuant with given config....')
                model, encoding_path = self._apply_autoquant()
                self.quantsim_config.model = model
                sim = self.get_qsim()
                self._raise_error_for_stateful_op()
                torch_encoding_path = encoding_path[:-10]+"_torch.encodings"
                sim.load_encodings(torch_encoding_path, strict=False, partial=True,
                                   requires_grad=False, allow_overwrite=False)
        else:
            sim = self.get_qsim()
            # Modify quantizer setting for activations and params if required based on cmd line args
            sim = self.quantsim_config.modify_quantizers_if_needed(sim)
            self._check_and_apply_stateful_op_exception_rules(sim)
            logger.info(f'Using "{self.quantsim_config.act_quant_scheme}" as activation quant calibration scheme '
                        f'and "{self.quantsim_config.param_quant_scheme}" as param quant calibration scheme')
            if not self.opts.ignore_encodings:
                if quantization_overrides:
                    logger.info('Quantization overrides provided, AIMET will compute any missing encodings')
                    self._raise_error_for_stateful_op()
                    sim.load_and_freeze_encodings(mpp_aligned_torch_enc_file, ignore_when_quantizer_disabled=True)
                else:
                    logger.info('No quantization overrides provided, AIMET will compute the full encodings')
            else:
                logger.info('--ignore_encodings flag is provided, AIMET will ignore any encodings provided')
            # Compute Encodings
            sim.compute_encodings(self.quantsim_config.forward_pass_fxn,
                                  (self.quantsim_config.dataloader, self._device,
                                   self.quantsim_config.iteration_size))

        ### EXPORT ####
        self._export(sim)

        if self.enable_backend_aware:
            quantsim_config.ENFORCE_TARGET_DTYPE_BITWIDTH_CONFIG = False

        weight_path = os.path.join(self.output_dir, self.output_name + '.safetensors')
        encoding_path = os.path.join(self.output_dir, self.output_name + '.json')

        logger.info('Quantization using AIMET Quantizer is done!')

        return weight_path, encoding_path

    def _export(self, sim: QuantizationSimModel):
        """
        Exports the encoding file and the updated weight file in safetensor format.

        :param sim: Quantization Sim Object
        """

        sim.export_weights_to_safetensors(self.output_dir, self.output_name)
        sim.save_encodings_to_json(self.output_dir, self.output_name)
