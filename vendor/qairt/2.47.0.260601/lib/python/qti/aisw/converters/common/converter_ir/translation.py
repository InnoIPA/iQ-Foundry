# ==============================================================================
#
#  Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common import ir_optimizer
from qti.aisw.converters.common.graph_optimizer import OptimizationPassModeManager


class Translation(object):
    def __init__(self):
        self.indexed_methods = {}

    def apply_method(self, method_name, *args, **kwargs):
        return self.indexed_methods[method_name](*args, **kwargs)

    def register_method(self, method_name, method):
        self.indexed_methods[method_name] = method

    def has_indexed_method(self, method_name):
        return method_name in self.indexed_methods


class TranslationBank(object):
    def __init__(self):
        # string type name -> translation
        # the same value may exist for multiple keys.
        self.translations = {}
        self.optimization_pass_mode_manager = OptimizationPassModeManager.get_instance()

    def __get_translation(self, op_type):
        if op_type not in self.translations:
            raise KeyError("No translation registered for op type {}.".format(op_type))
        return self.translations[op_type]

    def is_cpp_ir_pass(self, method_name):
        """
        Return true or false based on whether a method is to be applied
        or not depending on CPP-IR optimization_pipeline
        """
        if method_name in self.optimization_pass_mode_manager.get_cpp_ir_pass_list():
            log_debug1("{} Optimization is disabled in Python IR", method_name)
            return True
        return False

    def apply_method_to_op(self, op_type, method_name, *args, **kwargs):
        """
        Runs the requested method for the given op
        :param method_name: name of the method to call
        :param op_type: the operation type used to query the translation bank
        :param args: required positional arguments that will be passed to method

        raises KeyError if method not found for the requested op_type

        """
        if(self.is_cpp_ir_pass(method_name)):
            return

        translation = self.__get_translation(op_type)
        if not translation.has_indexed_method(method_name):
            raise KeyError("Translation for '%s' does not define an indexed method '%s'" % (op_type, method_name))
        return translation.apply_method(method_name, *args, **kwargs)

    def apply_method_to_all_ops(self, method_name, graph, *args, fail_if_no_method=True, **kwargs):
        """
        Runs the requested method on all ops in the given graph. i.e loops through the nodes in graph, gets the
        corresponding translation class for node and runs the given method with node and graph as args
        :param method_name: name of the method to call
        :param graph: the IR Opgraph to traverse
        :param args: required positional arguments that will be passed to method
        :param fail_if_no_method: flag to decide whether to fail or pass if translation not found for an Op
        :param kwargs: keywords arguments to this function

        raises KeyError if translation not registered for an op or if method not found for an op_type translation
                        unless fail_if_no_method is set to False in the kwargs argument
        """
        if(self.is_cpp_ir_pass(method_name)):
            return

        for node in graph.list_nodes():
            if graph.is_node_present(node):  # this extra check is needed since an optimization applied to op below
                                            # can remove next node(s), so doing a dynamic check is needed
                if node.op.type in self.translations and \
                        self.__get_translation(node.op.type).has_indexed_method(method_name):
                    self.apply_method_to_op(node.op.type, method_name, node, graph, *args, **kwargs)
                else:
                    if fail_if_no_method:
                        if  node.op.type not in self.translations:
                            log_debug("No translation registered for op type {}. Unable to apply method {}.", node.op.type, method_name)
                            raise KeyError("Converter does not support '{}' op type".format(node.op.type))
                        else:
                            raise NotImplementedError("{} method not found for op type {}.".format(method_name, node.op.type))

    def apply_method_to_graph(self, method_name, graph, *args, **kwargs):
        """
        Runs the requested method on graph. i.e loops through the registered translation classes and runs the
        given method with the graph as argument
        :param method_name:  name of the method to call
        :param graph: the IR Opgraph to traverse
        :param args: required positional arguments that will be passed to method
        :param kwargs: keywords arguments to be used to pass fail_if_no_method to this function

        raises KeyError if method no found for an op_type translation unless fail_if_no_method is set to False
                        in the kwargs argument
        """
        if(self.is_cpp_ir_pass(method_name)):
            return

        for type_name, translation in self.translations.items():
            if translation.has_indexed_method(method_name):
                translation.apply_method(method_name, graph, *args)
            else:
                fail_if_no_method = kwargs.get('fail_if_no_method', True)
                if fail_if_no_method:
                    raise NotImplementedError("{} method not found for op type {}.".format(method_name,type_name))

    def register_translation(self, translation, *op_types):
        for op_type in op_types:
            if op_type in self.translations:
                raise KeyError("A translation is already registered for op type '%s'" % op_type)
            self.translations[op_type] = translation


# Translation base class to be used across converters to translate source framework to IR IROpGraph
class ConversionTranslationBase(Translation):
    # method keys
    ADD_INPUT_OP = "ADD_INPUT_OP"
    ADD_OP = "ADD_OP"
    INFER_SHAPE = "INFER_SHAPE"

    def __init__(self):
        Translation.__init__(self)
        self.register_method(self.ADD_OP, self.add_op)
        self.register_method(self.ADD_INPUT_OP, self.add_input_op)
        self.register_method(self.INFER_SHAPE, self.infer_output_shapes)

    def add_op(self, src_op, context, **kwargs):
        graph = context.ir_graph
        op = self.extract_parameters(src_op, context)
        if op is None:
            return
        input_names = self.extract_input_names(src_op, context)
        output_names = self.extract_output_names(src_op, context)
        node = graph.add(op, input_names, output_names)
        self.add_src_op_info(node.op.name, src_op, graph)
        self.insert_default_trace_info(src_op, node, context)
        return node

    def add_input_op(self, src_input, graph, **kwargs):
        raise NotImplementedError("add_input_op for {} not implemented ".format(str(self.__class__.__name__)))

    def add_src_op_info(self, node_name, src_op, graph):
        raise NotImplementedError("add_src_op_info for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_parameters(self, src_op, context, **kwargs):
        raise NotImplementedError("extract_parameters for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, context, **kwargs):
        raise NotImplementedError("extract_input_names for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_output_names(self, src_op, context, **kwargs):
        raise NotImplementedError("extract_output_names for {} not implemented ".format(str(self.__class__.__name__)))

    def insert_default_trace_info(self, src_op, node, converter_context):
        """
        Func to insert a default 1 to 1 translation trace data for an op.
        it is needed to be implemented for each framework.
        :param src_op: the src_op from framework
        :param node: the target instance OpNode to be insert trace infomation.
                     also its outputs related buffers will be insert trace infomation.
        :param converter_context: the frontend converter context instance
        """
        graph = converter_context.ir_graph
        if graph.enable_trace:
            raise NotImplementedError("insert_default_trace_info for {} not implemented ".format(str(self.__class__.__name__)))

    def insert_trace_info(self, target, target_source_info, converter_context):
        """
        Func to insert a specific trace info for an op or a tensor from frameworks source info.
        :param target: the target instance or list of instances of OpNode or Buffer
        :param target_source_info: a instance or a list of the target source info. each source info
                                   is a tuple which contains the source name and source type("op"/"tensor")
                                   e.x. [(framework_source_name, framework_source_type), ...]
        :param converter_context: the frontend converter context instance
        """
        graph = converter_context.ir_graph
        if graph.enable_trace:
            if not isinstance(target, list):
                target = [target]
            if not isinstance(target_source_info, list):
                target_source_info = [target_source_info]
            for tgt in target:
                graph.set_trace_info(tgt, target_source_info)

    def infer_output_shapes(self, op, input_shapes):
        return [input_shapes[0]]
