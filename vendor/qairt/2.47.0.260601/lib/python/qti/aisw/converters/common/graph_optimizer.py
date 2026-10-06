# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import sys
import yaml
from enum import Enum, auto
import argparse
import threading

try:
    from qti.aisw.converters.common import ir_optimizer
    from qti.aisw.converters.common.backend_awareness import BackendInfo
    from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
    from qti.aisw.converters.common.utils import validation_utils
    from qti.aisw.converters.common.utils.converter_utils import log_debug1
except ImportError as ie:
    print("Failed to find necessary optimization packages:")
    print(str(ie))
    print("Please ensure that $QNN_SDK_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

# supported optimization stages
class OptimizationStage(Enum):
    PreLayout = ir_optimizer.OptimizationStage.PreLayout
    PostLayout = ir_optimizer.OptimizationStage.PostLayout
    PreQuant = ir_optimizer.OptimizationStage.PreQuant
    PostQuant = ir_optimizer.OptimizationStage.PostQuant

# supported optimization modes
class OptPassMode(Enum):
    IrOptimizerMainline = auto()
    IrOptimizerExperimental = auto()
    IrOptimizerDisable = auto()
    DisablePostQuantOptimizations = auto()

# passMode: mode in which CPP-IR passes to be executed
# mainline: set of CPP-IR passes enabled in mainline by default
# expt: set of CPP-IR passes which are dev tested and kept under experimental flag and not enabled in mainline by default
# NOTE: Both mainline and expt are a map of PyIr optimizaion name-> CPP-IR optimization name

class OptimizationPassInfo:
    pass_name_mapping = {
        'FOLD_RESHAPES_INSTANCE3':'FoldReshapesInstance3',
        'FOLD_RESHAPES_INSTANCE2':'FoldReshapesInstance2',
        'FOLD_RESHAPES_INSTANCE1':'FoldReshapesInstance1',
        'SQUASH_TRANSPOSE_RESHAPE':'OptimizeTransposeReshape',
        'HANDLE_GATHER_NEGATIVE_INDICES':'HandleGatherNegativeIndices',
        'FOLD_MULTIPLE_TRANSPOSE':'OptimizeTransposes',
        'SINK_TRANSPOSE_BELOW_SUM':'SinkTransposeBelowSum',
        'SQUASH_MULTIPLE_PERMUTE':'SquashMultiplePermute',
        'SQUASH_RESHAPE':'SquashReshape',
        'MATCH_SPACETODEPTH':'MatchSpaceToDepth',
        'MATCH_DEPTHTOSPACE':'OptimizeDepthToSpace',
        'MATCH_GATHERND':'OptimizeGatherND',
        'MULTI_TIME_STEPS_GRU':'MultiTimeStepsGRU',
        'SQUASH_PAD':'SquashPad',
        'ALIGN_MATMUL_RANKS':'AlignMatmulRanks',
        'FOLD_SOFTMAX':'FoldSoftmax',
        'FOLD_CONCATS':'FoldConcats',
        'REPLACE_6D_OPERATION':'Replace6dOps',
        'ADJUST_NORM_OP_BUFFERS':'AdjustNormOpBuffers',
        'ADJUST_NORM_OP_FLATTEN_LAST_DIM':'AdjustNormOpFlattenLastDims',
        'EXPAND_SPARSE_OP_STRUCTURE': 'ExpandSparseOpStructure',
        'SQUASH_CONSTANT_INPUT_INSTANCE2':'SquashConstantInput',
        'REMOVE_CAST_IDENTITY':'RemoveCastIdentity',
        'FORCE_PRUNE_CAST':'ForcePruneCast',
        'INJECT_CAST_FOR_GATHER':'InjectCastForGather',
        'REMOVE_IDENTITY_INSTANCE2':'RemoveNoOps2',
        'REMOVE_IDENTITY_INSTANCE1':'RemoveNoOps1',
        'REMOVE_DISCONNECTED_NODES':'RemoveDisconnectedNodes',
        'PROCESS_CUSTOM_IO_QUANT_INFO':'ProcessCustomIOQuantInfo',
        '_permute_axis_to_last':'PermuteAxisToLast',
        'OPTIMIZE_NEG':'OptimizeNegation',
        'OPTIMIZE_NON_INPLACE_CONCAT':'OptimizeNonInplaceConcat',
        'MULTI_TIME_STEPS_LSTM':'MultiTimeStepsLSTM',
        'UNROLL_LSTM_TIME_STEPS':'UnrollLstm',
        'UNROLL_GRU_TIME_STEPS' : 'UnrollGruTimeSteps',
        'EXPAND_GRU_OP_STRUCTURE':'ExpandGruOpStructure',
        'MATCH_GELU_APPROX':'MatchGeluApprox',
        'EXPAND_LSTM_OP_STRUCTURE':'ExpandLstmOpStructure',
        'EXPAND_INVERSE':'ExpandInverse',
        'SQUASH_BATCHNORM_INSTANCE2':'SquashBatchnorm',
        'OPTIMIZE_DEPTH_REDUCE':'OptimizeDepthReduce',
        '_reduce_eltwisebin_transpose':'ReduceEltwiseBinTranspose',
        'OPTIMIZE_UNMATCHED_MATMUL':'OptimizeUnmatchedMatMul',
        '_replace_trt_with_reshape':'ReplaceTRTWithReshape',
        '_squash_constant_transpose':'SquashConstantTranspose',
        '_fallback_depth_broadcast_eltwisebin':'FallbackDepthBroadcastEltwiseBin',
        '_optimize_depth_broadcast_elementwise':'OptimizeDepthBroadcastElementwise',
        'HOIST_TRANSPOSE':'HoistTranspose',
        '_remove_inverse_transpose_pair': 'RemoveInverseTransposePair',
        'expand_elementwise_mean':'ExpandElementwiseMean'
    }

    passes_map = {
        'mainline_passes': {
            'ENABLED': [
                'SINK_TRANSPOSE_BELOW_SUM',
                'SQUASH_MULTIPLE_PERMUTE',
                'OPTIMIZE_NEG',
                'REMOVE_IDENTITY_INSTANCE1',
                'FOLD_CONCATS',
                'FOLD_MULTIPLE_TRANSPOSE',
                'SQUASH_TRANSPOSE_RESHAPE',
                'SQUASH_BATCHNORM_INSTANCE2',
                'MATCH_GELU_APPROX',
                'MULTI_TIME_STEPS_LSTM',
                # If EXPAND_LSTM_OP_STRUCTURE/UNROLL_LSTM_TIME_STEPS is enabled, make sure
                # MULTI_TIME_STEPS_LSTM is also enabled. as
                # MULTI_TIME_STEPS_LSTM/Translation changes are required for EXPAND_LSTM/UNROLL_LSTM.
                'EXPAND_LSTM_OP_STRUCTURE',
                'UNROLL_LSTM_TIME_STEPS',
                'FOLD_RESHAPES_INSTANCE3',
                'MULTI_TIME_STEPS_GRU',
                # Please enable/disable UnrollGruTimeSteps and ExpandGruOpStructure passes together as
                # they are dependent on each other
                'UNROLL_GRU_TIME_STEPS',
                'EXPAND_GRU_OP_STRUCTURE',
                'HANDLE_GATHER_NEGATIVE_INDICES',
                'INJECT_CAST_FOR_GATHER',
                'REMOVE_IDENTITY_INSTANCE2',
                'EXPAND_SPARSE_OP_STRUCTURE',
                'PROCESS_CUSTOM_IO_QUANT_INFO',
                'SQUASH_CONSTANT_INPUT_INSTANCE2',
                'REMOVE_DISCONNECTED_NODES',
                'REPLACE_6D_OPERATION'
            ],
            'DISABLED': [
                'EXPAND_INVERSE',
                'OPTIMIZE_DEPTH_REDUCE',
                'OPTIMIZE_NON_INPLACE_CONCAT',
                'ALIGN_MATMUL_RANKS',
                'MATCH_GATHERND',
                'SQUASH_PAD',
                'SQUASH_RESHAPE',
                'FOLD_RESHAPES_INSTANCE2',
                'FOLD_RESHAPES_INSTANCE1',
                '_reduce_eltwisebin_transpose',
                '_permute_axis_to_last',
                'OPTIMIZE_UNMATCHED_MATMUL',
                '_replace_trt_with_reshape',
                'ADJUST_NORM_OP_BUFFERS',
                # ADJUST_NORM_OP_FLATTEN_LAST_DIM is dummy method created to maintain 1-to-1 mapping with G2G
                # so both should be enabled OR disabled together
                'ADJUST_NORM_OP_FLATTEN_LAST_DIM',
                '_squash_constant_transpose',
                '_fallback_depth_broadcast_eltwisebin',
                '_optimize_depth_broadcast_elementwise',
                'REMOVE_CAST_IDENTITY',
                'FORCE_PRUNE_CAST',
                'HOIST_TRANSPOSE',
                '_remove_inverse_transpose_pair',
                'expand_elementwise_mean'
            ]
        },
        'expt_passes': {
            'ENABLED': [
            ],
            'DISABLED': [
                'MATCH_SPACETODEPTH',
                'MATCH_DEPTHTOSPACE',
                'FOLD_SOFTMAX',
            ]
        }
    }


class OptimizationPassModeManager:

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
           with cls._lock:
                # Recheck for second instance in case of race condition
               if cls._instance is None:
                   cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not OptimizationPassModeManager._initialized:
            OptimizationPassModeManager._initialized = True
            # required_cpp_ir_pass_list list is only to enable passes in QNN/SNPE converters
            self.required_cpp_ir_pass_list = {'Replace6dOps'}
            # Need to be updated as per pass addition
            # Developers while adding a pass need to add it to the expt_default list
            # once verified by QA in a batch, those passes can be moved to mainline_default
            # by raising a new gerrit
            self.mainline_default, self.expt_default = self.format_optimization_map(OptimizationPassInfo.passes_map)
            # These are the passes after modified by config
            self.mainline_final = self.mainline_default.copy()
            self.expt_final = self.expt_default.copy()
            # Default optimization pass mode should be Python optimizations,
            # as in absense of any flags passes Python cases will fail otherwise
            self.passMode = OptPassMode.IrOptimizerDisable
            self.cpp_ir_passes_list = None
            self.recomputePassesList = True
            self.qairt_converter = False
            self.args = None

    def set_qairt_converter(self, qairt_converter):
        self.qairt_converter = qairt_converter

    def set_args(self, args):
        self.args = args

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # Call the Init
            OptimizationPassModeManager()
        return cls._instance

    def format_optimization_map(self, passes_map):
        mainline_passes = {}
        expt_passes = {}
        pass_name_mapping = OptimizationPassInfo.pass_name_mapping
        try:
            for p in passes_map['mainline_passes']['ENABLED']:
                mainline_passes[p] = [pass_name_mapping[p], True]
            for p in passes_map['mainline_passes']['DISABLED']:
                mainline_passes[p] = [pass_name_mapping[p], False]
            for p in passes_map['expt_passes']['ENABLED']:
                expt_passes[p] = [pass_name_mapping[p], True]
            for p in passes_map['expt_passes']['DISABLED']:
                expt_passes[p] = [pass_name_mapping[p], False]
        except:
            raise Exception(f"Pass {p} not found in Python-CPP_IR pass name mapping")
        return mainline_passes, expt_passes

    def get_optimization_pass_mode(self,passMode):
        if(passMode=="ir_optimizer_mainline"):
            return OptPassMode.IrOptimizerMainline
        elif(passMode=="ir_optimizer_experimental"):
            return OptPassMode.IrOptimizerExperimental
        elif(passMode=="disable_post_quant_optimizations"):
            return OptPassMode.DisablePostQuantOptimizations
        else:
            return OptPassMode.IrOptimizerDisable

    def set_pass_mode(self, args):
        self.recomputePassesList = True
        if hasattr(args, 'optimization_pass_mode') and args.optimization_pass_mode != "":
            self.passMode = self.get_optimization_pass_mode(args.optimization_pass_mode)
        else:
            self.passMode = OptPassMode.IrOptimizerDisable


    def parse_optimization_pass_mode_config(self, args):
        """
        This method reads the YAML config file and assigns the config to optimizer_context
        :return: None
        """
        if hasattr(args, 'optimization_pass_mode_config') and args.optimization_pass_mode_config != "":
            if hasattr(args, 'dump_optimization_pass_mode_config') and args.dump_optimization_pass_mode_config != False:
                raise Exception("dump_optimization_pass_mode_config and optimization_pass_mode_config cannot be passed together")

            pass_mode_config = args.optimization_pass_mode_config

            with open(pass_mode_config, 'r') as stream:
                try:
                    yaml_data = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    raise RuntimeError('Error parsing YAML config file') from e

            mainline_passes, expt_passes = self.format_optimization_map(yaml_data)

            # verify mainline_passes and expt_passes are subset of the mainline+expt lists
            for p in {*mainline_passes, *expt_passes}:
                if p not in {**self.mainline_final, **self.expt_final}:
                    raise Exception(f"Pass {p} not part of Mainline and Experimental passes. Passes should be part of: ",{*self.mainline_final, *self.expt_final})

            if {*mainline_passes} & {*expt_passes}:
                raise Exception("Mainline and Experimental Passes should NOT be overlapping")
            # Update the pipeline list to be processed
            self.mainline_final = mainline_passes
            self.expt_final = expt_passes

    def enable_pass(self, passName, doEnable = False):
        if passName in self.cpp_ir_passes_list:
            if not doEnable:
                self.cpp_ir_passes_list.discard(passName)
        else:
            if doEnable:
                self.cpp_ir_passes_list.add(passName)

    def configure_conditional_python_passes(self):
        """
        Adjusts the cpp_ir_passes_list based on LSTM-related arguments and the target backend.

        Rules (evaluated in priority order):
        1. If the backend is LPAI, or the user requests multi-time-step LSTM/GRU unrolling
           (multi_time_steps_lstm / multi_time_steps_gru), disable all three LSTM passes:
               UNROLL_LSTM_TIME_STEPS, UNROLL_LSTM_FOLD_RESHAPES, EXPAND_LSTM_OP_STRUCTURE

        2. Else if expand_lstm_op_structure is requested:
           - If EXPAND_LSTM_OP_STRUCTURE is already scheduled (in cpp_ir_passes_list) but
             MULTI_TIME_STEPS_LSTM is not yet scheduled, add MULTI_TIME_STEPS_LSTM so that
             the multi-time-step translation change required by EXPAND_LSTM_OP_STRUCTURE is
             also applied in CPP-IR.
           - Disable UNROLL_LSTM_TIME_STEPS and UNROLL_LSTM_FOLD_RESHAPES because they are
             mutually exclusive with the expand path.

        3. Else if unroll_lstm_time_steps is requested:
           - If UNROLL_LSTM_TIME_STEPS is already scheduled but MULTI_TIME_STEPS_LSTM is not,
             add MULTI_TIME_STEPS_LSTM (the translation change required by UNROLL_LSTM_TIME_STEPS).
           - Disable EXPAND_LSTM_OP_STRUCTURE because it is mutually exclusive with the unroll
             path.

        Note: MULTI_TIME_STEPS_LSTM and UNROLL_LSTM_TIME_STEPS / EXPAND_LSTM_OP_STRUCTURE must
        always be kept in the same enabled/disabled state (see the comment in passes_map).
        """
        # Nothing to do if args have not been set yet (e.g. during early initialisation).
        if self.args is None or not(self.qairt_converter):
            return

        # Condition 1: backend is LPAI, or multi-time-step LSTM/GRU mode is active.
        # In these cases the Python IR handles LSTM unrolling, so all three CPP-IR LSTM
        # passes must be disabled to avoid double-processing.
        if ((getattr(self.args, "backend", "HTP") and \
                getattr(self.args, "backend", "HTP").lower() == "lpai") or
                getattr(self.args, "multi_time_steps_lstm", False) or
                getattr(self.args, "multi_time_steps_gru", False)):
            self.enable_pass("UNROLL_LSTM_TIME_STEPS", doEnable=False)
            self.enable_pass("FOLD_RESHAPES_INSTANCE3", doEnable=False)
            self.enable_pass("EXPAND_LSTM_OP_STRUCTURE", doEnable=False)

        # Condition 2: user explicitly requests the expand-LSTM-op-structure path.
        elif getattr(self.args, "expand_lstm_op_structure", False):
            if "EXPAND_LSTM_OP_STRUCTURE" in self.cpp_ir_passes_list:
                if "MULTI_TIME_STEPS_LSTM" not in self.cpp_ir_passes_list:
                    # MULTI_TIME_STEPS_LSTM provides the translation changes that
                    # EXPAND_LSTM_OP_STRUCTURE depends on; enable it if not already present.
                    self.enable_pass("MULTI_TIME_STEPS_LSTM", doEnable=True)
            else:
                # if EXPAND_LSTM_OP_STRUCTURE is disabled(not in cpp_ir_passes_list)
                # then disable MULTI_TIME_STEPS_LSTM/Translation changes also
                self.enable_pass("MULTI_TIME_STEPS_LSTM", doEnable=False)

            # Unroll path is mutually exclusive with the expand path.
            self.enable_pass("UNROLL_LSTM_TIME_STEPS", doEnable=False)
            self.enable_pass("FOLD_RESHAPES_INSTANCE3", doEnable=False)

        # Condition 3: user explicitly requests the unroll-LSTM-time-steps path.
        elif getattr(self.args, "unroll_lstm_time_steps", False):
            if "UNROLL_LSTM_TIME_STEPS" in self.cpp_ir_passes_list:
                if "MULTI_TIME_STEPS_LSTM" not in self.cpp_ir_passes_list:
                    # MULTI_TIME_STEPS_LSTM provides the translation changes that
                    # UNROLL_LSTM_TIME_STEPS depends on; enable it if not already present.
                    self.enable_pass("MULTI_TIME_STEPS_LSTM", doEnable=True)
            else:
                # if UNROLL_LSTM_TIME_STEPS is disabled(not in cpp_ir_passes_list)
                # then disable MULTI_TIME_STEPS_LSTM/Translation changes also
                self.enable_pass("MULTI_TIME_STEPS_LSTM", doEnable=False)

            # Expand path is mutually exclusive with the unroll path.
            self.enable_pass("EXPAND_LSTM_OP_STRUCTURE", doEnable=False)

        is_unroll_gru_enabled = "UNROLL_GRU_TIME_STEPS" in self.cpp_ir_passes_list
        is_expand_gru_enabled = "EXPAND_GRU_OP_STRUCTURE" in self.cpp_ir_passes_list
        is_unroll_or_expand_arg_true =  not((getattr(self.args, "backend", "HTP") and \
                getattr(self.args, "backend", "HTP").lower() == "lpai") or
                getattr(self.args, "multi_time_steps_lstm", False) or
                getattr(self.args, "multi_time_steps_gru", False))

        if (is_unroll_or_expand_arg_true):
            if is_unroll_gru_enabled and is_expand_gru_enabled:
                self.enable_pass("MULTI_TIME_STEPS_GRU", doEnable=True)
            else:
                self.enable_pass("MULTI_TIME_STEPS_GRU", doEnable=False)
                self.enable_pass("EXPAND_GRU_OP_STRUCTURE", doEnable=False)
                self.enable_pass("UNROLL_GRU_TIME_STEPS", doEnable=False)
        else:
            self.enable_pass("EXPAND_GRU_OP_STRUCTURE", doEnable=False)
            self.enable_pass("UNROLL_GRU_TIME_STEPS", doEnable=False)

        """
        The Python ADJUST_NORM_OP_BUFFERS pass is split into two passes:
        1. ADJUST_NORM_OP_BUFFERS
        2. ADJUST_NORM_OP_FLATTEN_LAST_DIM (dummy pass for Python)

        +--------------------------------------------------+------------------------------------------+
        | Python                                           | G2G                                      |
        +--------------------------------------------------+------------------------------------------+
        | ADJUST_NORM_OP_BUFFERS                           | AdjustNormOpBuffers                      |
        | (operates on Transpose and Reshape)              | (operates on Transpose)                  |
        +--------------------------------------------------+------------------------------------------+
        | ADJUST_NORM_OP_FLATTEN_LAST_DIM (Dummy Pass)     | AdjustNormOpFlattenLastDims              |
        |                                                  | (operates on Reshape)                    |
        +--------------------------------------------------+------------------------------------------+

        To maintain functional correctness, ADJUST_NORM_OP_BUFFERS and ADJUST_NORM_OP_FLATTEN_LAST_DIM
        must be enabled or disabled together.
        """
        is_adjust_norm_op_buffer_enabled = "ADJUST_NORM_OP_BUFFERS" in self.cpp_ir_passes_list
        is_adjust_norm_op_flatten_enabled = "ADJUST_NORM_OP_FLATTEN_LAST_DIM" in self.cpp_ir_passes_list
        if is_adjust_norm_op_buffer_enabled != is_adjust_norm_op_flatten_enabled:
            raise Exception(
                "ADJUST_NORM_OP_BUFFERS and ADJUST_NORM_OP_FLATTEN_LAST_DIM must be enabled or disabled together. "
                f"Current state: ADJUST_NORM_OP_BUFFERS="
                f"{'enabled' if is_adjust_norm_op_buffer_enabled else 'disabled'}, "
                f"ADJUST_NORM_OP_FLATTEN_LAST_DIM="
                f"{'enabled' if is_adjust_norm_op_flatten_enabled else 'disabled'}."
            )

        # Check if the standard cast removal pass is enabled
        is_remove_cast_identity_pass_enabled = "REMOVE_CAST_IDENTITY" in self.cpp_ir_passes_list
        is_force_prune_cast_pass_enabled = "FORCE_PRUNE_CAST" in self.cpp_ir_passes_list

        # Validate that FORCE_PRUNE_CAST is not enabled without REMOVE_CAST_IDENTITY
        if is_force_prune_cast_pass_enabled and not is_remove_cast_identity_pass_enabled:
            raise Exception("FORCE_PRUNE_CAST is enabled but REMOVE_CAST_IDENTITY is disabled. "
                        "FORCE_PRUNE_CAST is designed to be used with REMOVE_CAST_IDENTITY. "
                        "Consider enabling REMOVE_CAST_IDENTITY or disabling FORCE_PRUNE_CAST.")


        # Enable the aggressive FORCE_PRUNE_CAST pass only if both conditions are met:
        # 1. The standard cast removal pass (REMOVE_CAST_IDENTITY) is enabled
        # 2. The user has explicitly requested aggressive cast pruning via force_prune_cast_ops flag
        # Otherwise, disable the aggressive pass to prevent unwanted data type changes
        if(is_remove_cast_identity_pass_enabled and getattr(self.args, "force_prune_cast_ops", False)):
            self.enable_pass("FORCE_PRUNE_CAST", doEnable=True)
            self.enable_pass("REMOVE_CAST_IDENTITY", doEnable=False)
        else:
            self.enable_pass("FORCE_PRUNE_CAST", doEnable=False)


    def set_cpp_ir_pass_list(self):
        """
            Create a list of methods to be skipped based on CPP-IR optimization_pipeline
        """
        self.cpp_ir_passes_list = set()
        # If a pass is required pass/ passName entry in self.required_cpp_ir_pass_list (e.g., 'Replace6dOps'),
        # it will ALWAYS be skipped in Python, regardless of the converter type.
        # If qairt_converter is True, ALL enabled mainline and experimental passes are skipped in Python and applied in G2G
        # Skip optimizations part of mainline pipeline if flag passed is ['ir_optimizer_mainline' or 'ir_optimizer_expt' or 'disable_post_quant_opt']
        if (self.passMode==OptPassMode.IrOptimizerMainline or
            self.passMode==OptPassMode.IrOptimizerExperimental or
            self.passMode==OptPassMode.DisablePostQuantOptimizations):
            self.cpp_ir_passes_list = set([pyPass for pyPass, cppPass in self.mainline_final.items() if cppPass[1] and (cppPass[0] in self.required_cpp_ir_pass_list or self.qairt_converter)])
        # Additionally skip experimental passes if flag passed is 'ir_optimizer_expt'
        if (self.passMode==OptPassMode.IrOptimizerExperimental):
            self.cpp_ir_passes_list.update(set([pyPass for pyPass, cppPass in self.expt_final.items() if cppPass[1] and (cppPass[0] in self.required_cpp_ir_pass_list or self.qairt_converter)]))

    def get_cpp_ir_pass_list(self):
        """
            Return a list of methods to be skipped based on CPP-IR optimization_pipeline
        """
        if self.recomputePassesList:
            self.set_cpp_ir_pass_list()
            self.configure_conditional_python_passes()
            self.recomputePassesList = False
        return self.cpp_ir_passes_list

class GraphOptimizer(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(GraphOptimizer.ArgParser, self).__init__(**kwargs)
            graph_optimizer_group = self.add_argument_group(title='Graph Optimizer Options')
            graph_optimizer_group.add_argument('--ir_optimizer_config',type=str,
                                                action=validation_utils.validate_filename_arg(must_exist=True),
                                                help=argparse.SUPPRESS, default="")
            graph_optimizer_group.add_argument('--dump_ir_optimizer_config_template', default=False, action="store_true",
                                                help=argparse.SUPPRESS)
            graph_optimizer_group.add_argument('--dump_pass_trace_info', default=False, action="store_true",
                                                help=argparse.SUPPRESS)
            graph_optimizer_group.add_argument('--dump_ir', nargs='+', type=str, default="",
                                                help=argparse.SUPPRESS)
            # Flag to select one of the pipelines among: ir_optimizer_mainline: Run mainline passes list and Disable corresponding passes in python,
            #                                            ir_optimizer_experimental: Run mainline+expt passes list and Disable corresponding passes in python,
            #                                            ir_optimizer_disable: Run all passes in python ONLY,
            #                                            disable_post_quant_optimizations: Disable all passes in python and G2G
            graph_optimizer_group.add_argument('--optimization_pass_mode', nargs='?', type=str, default="ir_optimizer_mainline",
                                                help=argparse.SUPPRESS)
            # Flag to pass config to manage passes in a pipeline
            graph_optimizer_group.add_argument("--optimization_pass_mode_config",type=str,
                                        action=validation_utils.validate_filename_arg(must_exist=True),
                                        help=argparse.SUPPRESS, default="")
            # Dump the hardcoded passes in the pipelines
            graph_optimizer_group.add_argument('--dump_optimization_pass_mode_config', default=False, action="store_true",
                                                help=argparse.SUPPRESS)

        @classmethod
        def convert_args(cls, args):
            args_dict = vars(args).copy()
            return args_dict

    def __init__(self, args):
        self.dump_ir_optimizer_config_template = args.get('dump_ir_optimizer_config_template', False)
        # set the default backend value to HTP if not specified, so that HTP pipeline will always trigger for QNN/SNPE converters
        self.backend_info_obj = BackendInfo.get_instance(args.get('backend') or "HTP", args.get('soc_model') or "")
        self.optimizer_ctx = ir_optimizer.IrOptimizerContext()
        self.optimizer_ctx.dump_pass_trace_info = args.get('dump_pass_trace_info', False)
        self.optimizer_ctx.enable_trace = True if args.get("enable_framework_trace") else False
        self.optimization_pass_mode_manager = OptimizationPassModeManager.get_instance()
        self.qairt_converter = self.optimization_pass_mode_manager.qairt_converter
        self.required_cpp_ir_pass_list = self.optimization_pass_mode_manager.required_cpp_ir_pass_list
        self.debug_log = True if (args.get('debug', -1) != -1) else False
        self.remove_unused_inputs = args.get('remove_unused_inputs', False)
        self.keep_disconnected_nodes = args.get("keep_disconnected_nodes")

        # Dump ir optimizer config template
        if self.dump_ir_optimizer_config_template:
            self.dump_ir_optimizer_config_yaml_template()
            sys.exit(0)

        # Parse ir optimizer config
        if args.get('ir_optimizer_config', "") != "":
            self.parse_optimizer_config(args['ir_optimizer_config'])

        # Configure conditional passes
        self.configure_conditional_passes(args)

        # Apply pipeline mode based optimization filter
        self.apply_optimization_pass_mode(args)

        # Filter base on qairt_converter flag
        self.filter_optimization_pass_for_targets(args)

        # Validate and initialize dump ir options.
        if args.get('dump_ir', "") != "":
            # Check the first value in the list to detect the Ir dump level.
            if args['dump_ir'][0] not in ['OPTIMIZER', 'PASS', 'OP', 'CHECKPOINT']:
                raise Exception("Invalid Ir dump level passed, supported values"
                                "are, OPTIMIZER, PASS, OP and CHECKPOINT")

            # Validate if pass names are passed only for PASS debugging.
            if args['dump_ir'][0] in ['OPTIMIZER', 'CHECKPOINT'] and len(args['dump_ir']) != 1:
                raise Exception("Passing individual pass names is only supported with"
                                "PASS dump")

            # Enable dump Ir option
            self.optimizer_ctx.dump_ir = True
            if args['dump_ir'][0] == 'OPTIMIZER':
                self.optimizer_ctx.dump_ir_level = ir_optimizer.IrDumpLevel.PreAndPostOptimization

            if args['dump_ir'][0] == 'PASS':
                self.optimizer_ctx.dump_ir_level = ir_optimizer.IrDumpLevel.PostPassChange
                if len(args['dump_ir']) != 1:
                    self.optimizer_ctx.dump_ir_pass_names = args['dump_ir'][1].split(",")
                    # Check whether the pass names passed in dump_ir is valid pass names.
                    registered_pass_names = ir_optimizer.get_registered_pass_names()
                    for pass_name in self.optimizer_ctx.dump_ir_pass_names:
                        if pass_name not in registered_pass_names:
                            raise Exception("Invalid pass name, ",pass_name," passed with"
                                            "dump_ir. No such pass is registered")

            if args['dump_ir'][0] == "OP":
                self.optimizer_ctx.dump_ir_level = ir_optimizer.IrDumpLevel.PostOpChange
                # Op level dumping requires mandatory operator types to be passed.
                if len(args['dump_ir']) == 1:
                    raise Exception("dump_ir OP level debugging requires operator "
                                    "types to passed.")

                self.optimizer_ctx.dump_ir_op_types = args['dump_ir'][1].split(",")

            if args['dump_ir'][0] == 'CHECKPOINT':
                self.optimizer_ctx.dump_ir_level = ir_optimizer.IrDumpLevel.Checkpoint

        # Validate optimization pipeline flag
        if args.get('optimization_pass_mode', "") != "":
            if args['optimization_pass_mode'] not in ['ir_optimizer_mainline', 'ir_optimizer_experimental', 'ir_optimizer_disable', 'disable_post_quant_optimizations']:
                raise Exception("Invalid optimization pipeline passed, supported values "
                                "ir_optimizer_mainline, ir_optimizer_experimental, ir_optimizer_disable, disable_post_quant_optimizations")

        # Dump optimization_pass_mode config
        if args.get('dump_optimization_pass_mode_config', False):
            self.dump_optimization_pass_mode_config()
            sys.exit(0)

        # If --dumpIR is passed, enable dump_ir with CHECKPOINT level to dump the full IrGraph
        # as JSON at key fixed points in the pass pipeline using IrGraph::dumpJson().
        if args.get('dumpIR', False):
            self.optimizer_ctx.dump_ir = True
            self.optimizer_ctx.dump_ir_level = ir_optimizer.IrDumpLevel.Checkpoint

        # Set debug log
        self.optimizer_ctx.debug_log = self.debug_log

    def parse_optimizer_config(self, optimizer_config):
        """
        This method reads the YAML config file and assigns the config to optimizer_context
        :return: None
        """
        with open(optimizer_config, 'r') as stream:
            try:
                yaml_data = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise RuntimeError('Error parsing YAML config file') from e

        opt_config = {"enabledPasses":[], "disabledPasses":[]}

        # Not possible to directly update self.optimizer_ctx.optimizer_config due to Pybind11
        # restriction. Create a dict and copy it to optimizer context.
        for key, value in yaml_data.items():
            for opt, status in value.items():
                if not status.get("skip"):
                    opt_config["enabledPasses"].append(opt)
                else:
                    opt_config["disabledPasses"].append(opt)

        self.optimizer_ctx.optimizer_config = opt_config

    def dump_ir_optimizer_config_yaml_template(self):
        """
        Dumps the yaml template for Ir optimizer configuration. This file can be edited
        as per the custom requirements and passed using the option ir_optimizer_config
        :return: None
        """
        yaml_data = {}
        # Get common pass config. backend_info None will return common optimization passes
        backend_info = None
        common_optimizations = ir_optimizer.get_dump_pass_config(backend_info)
        if len(common_optimizations):
            yaml_data["common"] = {}
            for pass_name in common_optimizations:
                yaml_data["common"][pass_name] = {"skip": common_optimizations[pass_name]}

        # Get backend pass config if backend arg is passed.
        if self.backend_info_obj:
            backend_info = self.backend_info_obj.get_backend_info_ptr()
            backend_optimizations = ir_optimizer.get_dump_pass_config(backend_info)
            if len(backend_optimizations):
                yaml_data["backend"] = {}
                for pass_name in backend_optimizations:
                    yaml_data["backend"][pass_name] = {"skip": backend_optimizations[pass_name]}

        with open("ir_optimizer_config_template.yaml", 'w', encoding="utf-8") as file:
            yaml.dump(yaml_data, file, sort_keys=False)

    def enablePass(self, config, passName, doEnable):
        if doEnable:
            if passName in config["disabledPasses"]:
                config["disabledPasses"].remove(passName)
            if passName not in config["enabledPasses"]:
                config["enabledPasses"].append(passName)
        else:
            if passName in config["enabledPasses"]:
                config["enabledPasses"].remove(passName)
            if passName not in config["disabledPasses"]:
                config["disabledPasses"].append(passName)

    def configure_conditional_passes(self, args):
        """
        This method disables/enables the conditional passes based on the pass arguments provided and updates the optimizer_context config.
        The individual conditional pass argument options take precedence over the options provided through the ir_optimizer_config.
        For instance, if pass P is enabled as per ir optimizer config but the argument enable_pass_P is false, the pass will be disabled.
        :return: None
        """
        backend_name = self.backend_info_obj.backend_name().lower() if self.backend_info_obj else ""
        opt_config = self.optimizer_ctx.optimizer_config
        opt_config.setdefault("enabledPasses", [])
        opt_config.setdefault("disabledPasses", [])

        self.enablePass(opt_config, "OptimizeGatherND", args["enable_match_gathernd"])
        self.enablePass(opt_config, "HandleGatherNegativeIndices", args["handle_gather_negative_indices"])
        self.enablePass(opt_config, "AlignMatmulRanks", args["align_matmul_ranks"])
        self.enablePass(opt_config, "ExpandSparseOpStructure", args["expand_sparse_op_structure"])
        # TODO: Register entry for REMOVE_DISCONNECTED_NODES, as it is also conditional Pass
        self.enablePass(opt_config, "ProcessCustomIOQuantInfo", args.get('io_config', False))
        if backend_name == "lpai" or \
            args["multi_time_steps_lstm"] or \
            args["multi_time_steps_gru"]:
            self.enablePass(opt_config, "UnrollGruTimeSteps", False)
            self.enablePass(opt_config, "ExpandGruOpStructure", False)
            self.enablePass(opt_config, "ExpandLstmOpStructure", False)
            self.enablePass(opt_config, "UnrollLstm", False)
            self.enablePass(opt_config, "FoldReshapesInstance3", False)
        # Disable UnrollLstm user pass expand_lstm_op_structure
        elif args["expand_lstm_op_structure"]:
            self.enablePass(opt_config, "UnrollLstm", False)
            self.enablePass(opt_config, "FoldReshapesInstance3", False)
        elif args["unroll_lstm_time_steps"]:
            self.enablePass(opt_config, "ExpandLstmOpStructure", False)
        self.enablePass(opt_config, "SquashBatchnorm", not args["disable_batchnorm_folding"])

        # AdjustNormOpFlattenLastDims is only enabled when backend is None or HTP
        enable_flatten_last_dim = (backend_name in ("", "htp"))
        self.enablePass(opt_config, "AdjustNormOpFlattenLastDims", enable_flatten_last_dim)

        self.optimizer_ctx.optimizer_config = opt_config

    def filter_optimization_pass_for_targets(self, args):
        opt_config = self.optimizer_ctx.optimizer_config
        opt_config.setdefault("enabledPasses", [])
        opt_config.setdefault("disabledPasses", [])
        enabled_set = opt_config["enabledPasses"].copy()

        for passName in enabled_set:
            # If it is QNN/SNPE converter and Pass is not required pass
            # Then disable that pass
            if not self.qairt_converter and passName not in self.required_cpp_ir_pass_list:
                self.enablePass(opt_config, passName, doEnable = False)

        self.optimizer_ctx.optimizer_config = opt_config

    def apply_optimization_pass_mode(self, args):
        """
        This method disables/enables the passes based on the pass mode arguments provided and updates the optimizer_context config.
        At this point ir_optimizer_config and condition_passes based filtering has been performed.
        Case a:
            If opt_pass_mode = ir_optimizer_disable OR disable_post_quant_optimizations
                ALL enabled_passes and mainline + expt passes will be moved to disabled --> Essentially no passes using CPP-IR will run
        Case b:
            If opt_pass_mode = ir_optimizer_mainline OR ir_optimizer_expt
                All passes in mainline pipeline (plus expt in case of ir_optimizer_expt) will be executed GIVEN it is not explicitely part of disable_passes already
                NOTE: Any extra passes already part of enabled_passes which is not part of mainline( plus expt in case of ir_optimizer_expt) WILL BE MOVED to disabled_passes

        One has to be CAUTIOUS while using this method along with ir_optimizer_config, because op_graph_optimization passes won't be enabled/disabled
        based on ir_optimizer_config.
        For Example:
            Pass "A" is part of mainline, but disabled with ir_optimizer_config.
            So "A" won't be run using op_graph_optimization, and also won't be run with CPP-IR -> causing a discrepancy between actual and expected.
        :return: None
        """
        opt_config = self.optimizer_ctx.optimizer_config
        opt_config.setdefault("enabledPasses", [])
        opt_config.setdefault("disabledPasses", [])

        opt_pass_mode = self.optimization_pass_mode_manager.passMode
        # Assuming there can be passes not part of mainline/expt for safety
        if(opt_pass_mode == OptPassMode.IrOptimizerDisable or
            opt_pass_mode == OptPassMode.DisablePostQuantOptimizations):
            disabled_set = set(opt_config["enabledPasses"])
            disabled_set.update([val[0] for val in self.optimization_pass_mode_manager.mainline_default.values()])
            disabled_set.update([val[0] for val in self.optimization_pass_mode_manager.expt_default.values()])
            disabled_set.update(opt_config["disabledPasses"])
            opt_config["enabledPasses"] = []
            opt_config["disabledPasses"] = list(disabled_set)

        elif(opt_pass_mode == OptPassMode.IrOptimizerMainline):
            # Remove those passes from mainline which are explicitely disabled
            enabled_set = set([val[0] for val in self.optimization_pass_mode_manager.mainline_final.values() if val[1]]) - set(opt_config["disabledPasses"])
            # Disable passes present in enabledPasses but not in mainline
            disabled_set = set(opt_config["enabledPasses"]) - enabled_set
            # Whichever passes disabled should still remain disabled
            disabled_set.update(opt_config["disabledPasses"])
            disabled_set.update([val[0] for val in self.optimization_pass_mode_manager.mainline_final.values() if not val[1]])
            # Any pass not part of mainline list to be disabled
            disabled_set.update((set([val[0] for val in self.optimization_pass_mode_manager.mainline_default.values()])|\
                                set([val[0] for val in self.optimization_pass_mode_manager.expt_default.values()]))\
                                 - enabled_set)
            opt_config["enabledPasses"] = list(enabled_set)
            opt_config["disabledPasses"] = list(disabled_set)

        elif(opt_pass_mode == OptPassMode.IrOptimizerExperimental):
            # Remove those passes from mainline + expt which are explicitely disabled
            enabled_set = (set([val[0] for val in self.optimization_pass_mode_manager.mainline_final.values() if val[1]])|\
                            set([val[0] for val in self.optimization_pass_mode_manager.expt_final.values() if val[1]])) -\
                             set(opt_config["disabledPasses"])
            # Disable passes present in enabledPasses but not in mainline + expt
            disabled_set = set(opt_config["enabledPasses"]) - enabled_set
            # Filtered out mainline and experimental passes using config should also be disabled
            disabled_set.update((set([val[0] for val in self.optimization_pass_mode_manager.mainline_default.values()])|\
                                set([val[0] for val in self.optimization_pass_mode_manager.expt_default.values()])) \
                                - enabled_set)
            # Whichever passes disabled should still remain disabled
            disabled_set.update(opt_config["disabledPasses"])
            disabled_set.update([val[0] for val in self.optimization_pass_mode_manager.mainline_final.values() if not val[1]])
            disabled_set.update([val[0] for val in self.optimization_pass_mode_manager.expt_final.values() if not val[1]])
            opt_config["enabledPasses"] = list(enabled_set)
            opt_config["disabledPasses"] = list(disabled_set)

        # TODO: currently RemoveDisconnectedNodes is not present in IrOptimizer as Pass
        # Remove this code once RemoveDisconnectedNodes is added as Pass
        # Filter out RemoveDisconnectedNodes from IrOptimizer passes list
        if "RemoveDisconnectedNodes" in opt_config["enabledPasses"]:
            opt_config["enabledPasses"].remove("RemoveDisconnectedNodes")
        if "RemoveDisconnectedNodes" in opt_config["disabledPasses"]:
            opt_config["disabledPasses"].remove("RemoveDisconnectedNodes")

        # Disable all passes except required_cpp_ir_pass_list for non QAIRT target
        if(not self.qairt_converter):
            backend_optimizations = set(ir_optimizer.get_registered_pass_names())
            passes_to_disable = backend_optimizations - self.required_cpp_ir_pass_list
            for pass_name in passes_to_disable:
                # Disable the pass
                log_debug1(f"{pass_name} is disabled in CPP IR")
                self.enablePass(opt_config, pass_name, doEnable=False)

        self.optimizer_ctx.optimizer_config = opt_config

    def dump_optimization_pass_mode_config(self):
        """
        Dumps the yaml template for optimization pipeline configuration. This file can be edited
        as per the custom requirements and passed using the option optimization_pass_mode_config
        :return: None
        """

        with open("optimization_pass_mode_config.yaml", 'w', encoding="utf-8") as file:
            comment_str = "# By Default ir_optimizer_mainline pipeline is used\n" \
                          "# Experimental passes can be tested in two way:\n" \
                          "# \t1. Pass ir_optimizer_experimental as the flag \n" \
                          "# \t2. Move the pass from experimental list to mainline line in this file\n\n"
            file.write(comment_str)
            yaml.dump(OptimizationPassInfo.passes_map, file, sort_keys=False)

    def optimize(self, ir_graph, optimization_stages:list):
        """
        This method optimizes the IR graph (inplace).
        :return: None
        """
        enabled_optimization_stages = [stage.value for stage in optimization_stages]
        backend_info = None
        # Create backed_info object if specific backend is passed to apply backend specific optimizations
        if self.backend_info_obj:
            backend_info = self.backend_info_obj.get_backend_info_ptr()
        optimizer = ir_optimizer.IrOptimizer(ir_graph, enabled_optimization_stages, backend_info, self.optimizer_ctx)

        # Debug log
        log_debug1("List of passes enabled in CPP IR")
        for opt_pass in self.optimizer_ctx.optimizer_config["enabledPasses"]:
            log_debug1(f"{opt_pass} Pass is enabled in IrOptimizer")

        # Apply optimizations(in place) on IrGraph
        optimizer.optimize()

        # TODO: Remove explicit call for REMOVE_DISCONNECTED_NODES from here
        # once REMOVE_DISCONNECTED_NODES is present as Pass in IrOptimizer
        if not self.keep_disconnected_nodes and "REMOVE_DISCONNECTED_NODES" in self.optimization_pass_mode_manager.get_cpp_ir_pass_list():
            ir_graph.remove_disconnected_nodes(self.remove_unused_inputs)

        # Pass execution summary
        log_debug1("Pass Execution Summary: ")
        log_debug1("\t Passes disabled in Python IR: ")
        for opt_pass in self.optimization_pass_mode_manager.get_cpp_ir_pass_list():
            log_debug1(f"\t\t{opt_pass} Optimization is disabled in Python IR")
        log_debug1("\n\t Optimization passes executed in order in CPP IR: ")
        for opt_pass in optimizer.get_current_execution_pass_names():
            if opt_pass in self.optimizer_ctx.optimizer_config["enabledPasses"]:
                log_debug1(f"\t\t {opt_pass} Pass is executed in IrOptimizer")
        log_debug1("\n\t Optimization passes Disabled in CPP IR: ")
        for opt_pass in self.optimizer_ctx.optimizer_config["disabledPasses"]:
            log_debug1(f"\t\t {opt_pass} Pass is disabled in IrOptimizer")
