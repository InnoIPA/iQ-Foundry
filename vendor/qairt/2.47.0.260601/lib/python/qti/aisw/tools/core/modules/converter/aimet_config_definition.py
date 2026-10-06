# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Optional, List, Any, Callable
from pydantic import Field, model_serializer

from qti.aisw.tools.core.modules.api import AISWBaseModel


class QuantSimConfig(AISWBaseModel):
    '''
    Config class for AIMET QuantSim
    '''
    dataloader: Optional[Any] = Field(default=None, description="Unlabeled Pytorch Dataloader for "
                                                                "QuantSim algorithm")
    iteration_size: Optional[int] = Field(default=None, description="Number of iterations of "
                                                                    "forward pass on dataloader "
                                                                    "provided.")

    @model_serializer
    def serialize_model(self) -> dict:

        _quantsim_args = {}
        _quantsim_args['dataloader'] = self.dataloader
        _quantsim_args['iteration_size'] = self.iteration_size
        return {'quantsim': _quantsim_args}


class AdaRoundConfig(AISWBaseModel):
    '''
    Config class for AIMET AdaRound API
    '''
    dataloader: Optional[Any] = Field(default=None, description="Unlabeled Pytorch Dataloader for "
                                                                "AdaRound algorithm")
    num_batches: Optional[int] = Field(default=None, description="Number of batches to be used for "
                                                                 "adaround iteration.")
    iteration_size: Optional[int] = Field(default=None, description="Number of iterations of "
                                                                 "forward pass in computing "
                                                                 "encodings")
    default_num_iterations: Optional[int] = Field(default=None, description="Number of iterations "
                                                                            "to adaround for each "
                                                                            "layer.The default "
                                                                            "value is 10K for "
                                                                            "models with 8- or "
                                                                            "higher bit weights, "
                                                                            "and 15K for models "
                                                                            "with lower than 8 bit "
                                                                            "weights.")
    default_reg_param: Optional[float] = Field(default=None, description="Regularization parameter"
                                                                         ", trading off between "
                                                                         "rounding loss vs "
                                                                         "reconstruction loss")
    default_beta_range: Optional[list] = Field(default=None, description="List of start and stop "
                                                                         "beta parameter for "
                                                                         "annealing of rounding loss "
                                                                         "[start_beta, end_beta]")
    default_warm_start: Optional[float] = Field(default=None, description="Warm up period, during "
                                                                          "which rounding loss has "
                                                                          "zero effect")
    forward_fn: Optional[Callable] = Field(default=None, description="Callback function that "
                                                                     "performs forward pass given "
                                                                     "a model and inputs yielded "
                                                                     "from the data loader. The "
                                                                     "function expects model as "
                                                                     "first argument and inputs to "
                                                                     "model as second argument")

    default_param_bw: Optional[int] = Field(default=None, ge=4, le=31, description="Default "
                                                                                   "bitwidth (4-31) "
                                                                                   "to use for "
                                                                                   "quantizing "
                                                                                   "layer "
                                                                                   "parameters")
    param_bw_override_list: Optional[list] = Field(default=None, description="List of lists, each "
                                                                             "list contains a  "
                                                                             "module and its "
                                                                             "corresponding "
                                                                             "parameter bitwidth. ")
    ignore_quant_ops_list: Optional[list] = Field(default=None, description="Ops listed here are "
                                                                            "skipped during "
                                                                            "quantization needed "
                                                                            "for AdaRounding. Do "
                                                                            "not specify Conv and "
                                                                            "Linear modules in "
                                                                            "this list. Doing so, "
                                                                            "will affect accuracy")
    default_quant_scheme: Optional[str] = Field(default=None, description="Quantization scheme. "
                                                                          "Supported options are "
                                                                          "post_training_tf or "
                                                                          "post_training_tf_enhanced")
    default_config_file: Optional[str] = Field(default=None, description="Default configuration "
                                                                         "file path for model "
                                                                         "quantizers")

    @model_serializer
    def serialize_model(self) -> dict:

        _adaround_args = {}
        _adaround_args['dataloader'] = self.dataloader
        _adaround_args['num_batches'] = self.num_batches
        _adaround_args['iteration_size'] = self.iteration_size

        optional_adaround_param_args = {}
        param_list = ['default_num_iterations', 'default_reg_param', 'default_beta_range',
                      'default_warm_start', 'forward_fn']
        for param in param_list:
            if getattr(self, param) is not None:
                optional_adaround_param_args[param] = getattr(self, param)
        _adaround_args['optional_adaround_param_args'] = optional_adaround_param_args
        optional_adaround_args = {}
        param_list = ['default_param_bw', 'param_bw_override_list', 'ignore_quant_ops_list',
                      'default_quant_scheme', 'default_config_file']
        for param in param_list:
            if getattr(self, param) is not None:
                optional_adaround_args[param] = getattr(self, param)
        _adaround_args['optional_adaround_args'] = optional_adaround_args
        return {'adaround': _adaround_args}


class AMPConfig(AISWBaseModel):
    '''
    Config class for AIMET AMP API
    '''
    dataloader: Any = Field(description="Labeled Pytorch Dataloader for AMP algorithm")
    candidates: List = Field(description="List of lists of candidate bitwidths and datatypes.")
    eval_callback_for_phase2: Callable[[Any, Any], float] = Field(description="evaluator function "
                                                                              "which takes "
                                                                              "predicted batch as "
                                                                              "the first argument "
                                                                              "and ground truth "
                                                                              "batch as the second "
                                                                              "argument and "
                                                                              "returns calculated "
                                                                              "metric as float "
                                                                              "value.")
    iteration_size: Optional[int] = Field(default=None, description="Number of iterations of "
                                                                    "forward pass in computing "
                                                                    "encodings")
    allowed_accuracy_drop: Optional[float] = Field(default=0.0, description="Maximum allowed drop "
                                                                            "in accuracy from "
                                                                            "FP32 baseline")
    eval_callback_for_phase1: Optional[Any] = Field(default=None, description="An object of "
                                                                              "CallbackFunc class "
                                                                              "(defined in AIMET)"
                                                                              "that takes "
                                                                              "in Eval function "
                                                                              "(callable) and "
                                                                              "eval function "
                                                                              "parameters. "
                                                                              "This "
                                                                              "evaluation function "
                                                                              "measures the "
                                                                              "sensitivity of each "
                                                                              "quantizer group "
                                                                              "during phase 1")
    forward_pass_callback: Optional[Any] = Field(default=None, description="An object of "
                                                                           "CallbackFunc class "
                                                                           "which takes in "
                                                                           "Forward pass function "
                                                                           "(callable) and its "
                                                                           "function parameters")
    use_all_amp_candidates: Optional[bool] = Field(default=None, description="When the field "
                                                                             "“use_all_amp_candidates” "
                                                                             "is set to True, the "
                                                                             "AMP algorithm will "
                                                                             "ignore the "
                                                                             "'supported_kernels' "
                                                                             "in the config file "
                                                                             "and continue to use "
                                                                             "all candidates")
    phase2_reverse: Optional[bool] = Field(default=None, description="If this parameter is set to "
                                                                     "True, phase1 of the AMP "
                                                                     "algorithm (calculating the "
                                                                     "accuracy list) remains "
                                                                     "unchanged. However, phase2 "
                                                                     "(generating the Pareto list) "
                                                                     "will be modified. In phase2, "
                                                                     "the algorithm starts with all "
                                                                     "quantizer groups in the least "
                                                                     "candidate model and "
                                                                     "incrementally moves nodes to "
                                                                     "higher candidates until the "
                                                                     "target accuracy is achieved")
    amp_search_algo: Optional[str] = Field(default=None, description="Defines the search algorithm "
                                                                     "to be used for the phase 2 "
                                                                     "of AMP. Supported Options : "
                                                                     "Binary, Interpolation and "
                                                                     "BruteForce")
    clean_start: Optional[bool] = Field(default=None, description="If true, any cached "
                                                                  "information from previous runs "
                                                                  "will be deleted prior to "
                                                                  "starting the mixed-precision "
                                                                  "analysis. If false, prior "
                                                                  "cached information will be "
                                                                  "used if applicable")

    @model_serializer
    def serialize_model(self) -> dict:

        _amp_args = {}
        _amp_args['dataloader'] = self.dataloader
        _amp_args['candidates'] = self.candidates
        _amp_args['eval_callback_for_phase2'] = self.eval_callback_for_phase2
        _amp_args['allowed_accuracy_drop'] = self.allowed_accuracy_drop
        _amp_args['iteration_size'] = self.iteration_size

        optional_amp_args = {}
        param_list = ['eval_callback_for_phase1', 'forward_pass_callback', 'use_all_amp_candidates',
                      'phase2_reverse', 'amp_search_algo', 'clean_start']
        for param in param_list:
            if getattr(self, param) is not None:
                optional_amp_args[param] = getattr(self, param)
        _amp_args['optional_amp_args'] = optional_amp_args

        return {'amp': _amp_args}


class AutoQuantConfig(AISWBaseModel):
    '''
    Config class for AIMET AutoQuant API
    This API integrate and apply post-training quantization techniques.

    AutoQuant includes 1) batchnorm folding, 2) cross-layer equalization,
    3) Adaround, and 4) Automatic Mixed Precision (if enabled).
    These techniques will be applied in a best-effort manner until the model
    meets the evaluation goal given as allowed_accuracy_drop.
    '''
    allowed_accuracy_drop: float = Field(default=0.0, description="Maximum allowed drop in "
                                                                  "accuracy from FP32 baseline")
    iteration_size: Optional[int] = Field(default=None, description="Number of iterations of "
                                                                    "forward pass in computing "
                                                                    "encodings")
    # AutoQuant Args
    param_bw: Optional[int] = Field(default=None, description="Parameter bitwidth.")
    output_bw: Optional[int] = Field(default=None, description="Output bitwidth.")
    quant_scheme: Optional[str] = Field(default=None, description="Quantization scheme")
    rounding_mode: Optional[str] = Field(default=None, description="Rounding mode")
    config_file: Optional[str] = Field(default=None, description="Path to configuration file for "
                                                                 "model quantizers")
    cache_id: Optional[str] = Field(default=None, description="ID associated with cache results")
    strict_validation: Optional[bool] = Field(default=None, description="Flag set to True by "
                                                                        "default.If False, "
                                                                        "AutoQuant will proceed "
                                                                        "with execution and handle "
                                                                        "errors internally if "
                                                                        "possible. This may "
                                                                        "produce unideal or "
                                                                        "unintuitive results.")
    # AdaRound Args
    dataloader: Any = Field(description="Unlabeled Pytorch Dataloader for AdaRound algorithm")
    num_batches: Optional[int] = Field(default=None, description="Number of batches to be used for "
                                                                 "adaround iteration.")
    default_num_iterations: Optional[int] = Field(default=None, description="Number of iterations "
                                                                            "to adaround each layer. "
                                                                            "The default value is "
                                                                            "10K for models with "
                                                                            "8- or higher bit "
                                                                            "weights, and 15K for "
                                                                            "models with lower than "
                                                                            "8 bit weights.")
    default_reg_param: Optional[float] = Field(default=None, description="Regularization parameter, "
                                                                         "trading off between "
                                                                         "rounding loss vs "
                                                                         "reconstruction loss")
    default_beta_range: Optional[list] = Field(default=None, description="List of start and stop "
                                                                         "beta parameter for "
                                                                         "annealing of rounding loss "
                                                                         "(start_beta, end_beta)")
    default_warm_start: Optional[float] = Field(default=None, description="Warm up period, during "
                                                                          "which rounding loss has "
                                                                          "zero effect")
    forward_fn: Optional[Callable] = Field(default=None, description="Callback function that "
                                                                     "performs forward pass given "
                                                                     "a model and inputs yielded "
                                                                     "from the data loader. The "
                                                                     "function expects model as "
                                                                     "first argument and inputs to "
                                                                     "model as second argument")
    # AMP Args
    amp_candidates: list = Field(description="List of possible bitwidths and datatypes for AMP")
    eval_callback: Callable = Field(description="Evaluator function which takes predicted value "
                                                "batch as the first argument and ground truth "
                                                "batch as the second argument and returns "
                                                "calculated metric float value.")
    eval_dataloader: Any = Field(description="Labeled Pytorch Dataloader for AdaRound algorithm")
    num_samples_for_phase_1: Optional[int] = Field(default=None, description="Number of samples to "
                                                                             "be used for "
                                                                             "performance "
                                                                             "evaluation in AMP "
                                                                             "phase 1")
    amp_forward_fn: Optional[Callable] = Field(default=None, description="Callback function that "
                                                                         "performs forward pass "
                                                                         "given a model and inputs "
                                                                         "yielded from the data "
                                                                         "loader. The function "
                                                                         "expects model as first "
                                                                         "argument and inputs to "
                                                                         "model as second argument")
    num_samples_for_phase_2: Optional[int] = Field(default=None, description="Number of samples to "
                                                                             "be used for "
                                                                             "performance "
                                                                             "evaluation in AMP "
                                                                             "phase 2")

    @model_serializer
    def serialize_model(self) -> dict:

        _autoquant_args = {}
        _autoquant_args['dataloader'] = self.dataloader
        _autoquant_args['allowed_accuracy_drop'] = self.allowed_accuracy_drop
        _autoquant_args['eval_callback'] = self.eval_callback
        _autoquant_args['eval_dataloader'] = self.eval_dataloader
        _autoquant_args['amp_candidates'] = self.amp_candidates
        _autoquant_args['iteration_size'] = self.iteration_size

        optional_autoquant_args = {}
        param_list = ['param_bw', 'output_bw', 'quant_scheme', 'rounding_mode', 'config_file',
                      'cache_id', 'strict_validation']
        for param in param_list:
            if getattr(self, param) is not None:
                optional_autoquant_args[param] = getattr(self, param)
        _autoquant_args['optional_autoquant_args'] = optional_autoquant_args

        optional_adaround_param_args = {}
        param_list = ['num_batches', 'default_num_iterations', 'default_reg_param',
                      'default_beta_range', 'default_warm_start', 'forward_fn']
        for param in param_list:
            if getattr(self, param) is not None:
                optional_adaround_param_args[param] = getattr(self, param)
        _autoquant_args['optional_adaround_args'] = optional_adaround_param_args

        optional_amp_args = {}
        param_list = ['num_samples_for_phase_1', 'num_samples_for_phase_2']

        for param in param_list:
            if getattr(self, param) is not None:
                optional_amp_args[param] = getattr(self, param)

        if getattr(self, 'amp_forward_fn') is not None:
            optional_amp_args['forward_fn'] = getattr(self, 'amp_forward_fn')

        _autoquant_args['optional_amp_args'] = optional_amp_args

        return {'autoquant': _autoquant_args}
