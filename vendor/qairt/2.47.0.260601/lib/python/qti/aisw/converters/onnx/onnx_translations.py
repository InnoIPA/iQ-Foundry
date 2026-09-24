# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import numpy as np
from abc import ABCMeta

from qti.aisw.converters.common.converter_ir import translation, op_adapter
from qti.aisw.converters.common.converter_ir.op_graph import TraceType
from qti.aisw.converters.onnx.op_schema import OpSchemaBase, OpSchemaDict, OP_SCHEMA_REGISTRY
from .util import *

import onnx
from onnx import defs, TensorProto

OnnxTranslations = translation.TranslationBank()
global ALL_SCHEMA_NAMES


class OnnxTranslationBase(translation.ConversionTranslationBase):
    # onnx specific translation method keys
    ADD_INPUT = "ADD_INPUT"
    SUPPORTED_VERSION = "SUPPORTED_VERSION"

    def __init__(self):
        global ALL_SCHEMA_NAMES

        translation.ConversionTranslationBase.__init__(self)
        self.register_method(self.SUPPORTED_VERSION, self.get_supported_version)
        # list of dictionary-style class that maps {version:op_schema_list}
        self._op_schemas = []
        ALL_SCHEMA_NAMES = [schema.name for schema in defs.get_all_schemas() if schema.domain == ""]

    @staticmethod
    def fetch_constant_op(name, converter_context, *, prunable=True, quantizable=None, dtype=None, fail_if_dynamic=True,
                          fail_if_not_found=False):
        """
        Gets ConstantOp object for given tensor name if static
        :param name: the name of the tensor to look up
        :param graph: the IROpgraph instance
        :param prunable: determines if ConstantOp associated with provided name will be consumed by its
                         consumer node and if so it will later be pruned
        :param quantizable: if the ConstantOp associated with name is quantizable
        :param dtype: the data type of the tensor to be fetched from the weights list
        :param fail_if_dynamic: flag if True will raise ValueError if given tensor name is dynamic
        :param fail_if_not_found: flag if True will raise ValueError if given tensor name is not found
        :return: the ConstantOp associated with the provided name if found, None otherwise
        :raises ValueError if buffer is found for provided name but is not produced by a ConstantOp
        """
        op = None
        graph = converter_context.ir_graph
        if not graph.has_buffer(name) and converter_context.weights.has(name):
            const_tensor = converter_context.weights.fetch_constant_tensor(name, prunable=prunable, dtype=dtype)
            # if caller has not set quantizable, determine based on data type
            if quantizable is None:
                quantizable = False
                if converter_context.weights.weight_map[name].dtype in [np.float16, np.float32, np.float64]:
                    quantizable = True
            op = op_adapter.ConstantOp(name, const_tensor, quantizable=quantizable)
        elif graph.has_buffer(name):
            op = graph.get_producer_op(name)
            if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                # reset quantizable property only if the caller has explicitly requested
                if quantizable is not None:
                    op.quantizable = quantizable
                # Constant Op translation adds the associated tensor to the weight map, use that to mark
                # prunability.
                if name in converter_context.weights.weight_map:
                    converter_context.weights.weight_map[name].consumed = prunable
            elif fail_if_dynamic:
                raise ValueError("Dynamic value for tensor name: {}, is not supported.".format(name))
            else:
                return None
        elif fail_if_not_found:
            raise ValueError("Input tensor: {} not found in the graph.".format(name))
        return op

    def add_src_op_info(self, node_name, src_op, graph):
        graph.add_src_op_info(node_name,
                              [i for i in src_op.input],
                              [o for o in src_op.output])

    def extract_parameters(self, src_op, converter_context):
        raise NotImplementedError("extract_parameters for {} not "
                                  "implemented ".format(str(self.__class__.__name__)))

    def extract_input_names(self, src_op, converter_context):
        return list(map(str, src_op.input))

    def extract_output_names(self, src_op, converter_context):
        return list(map(str, src_op.output))

    # insert trace information for one-to-one op translation
    def insert_default_trace_info(self, src_op, node, converter_context):
        graph = converter_context.ir_graph
        if graph.enable_trace:
            graph.set_trace_info(node, [(src_op.name, TraceType.OP)])
            for i, name in enumerate(node.output_names):
                buf = graph.get_buffer(name)
                graph.set_trace_info(buf, [(node.output_names[i], TraceType.TENSOR)])
            # consider the weights comming from exact op of framework model
            for i, name in enumerate(node.input_names):
                if name in converter_context.weights_trace_info:
                    trace_info = copy.deepcopy(converter_context.weights_trace_info[name])
                    if name in graph.trace_dict[TraceType.TENSOR]:
                        trace_info.extend(graph.trace_dict[TraceType.TENSOR][name])
                        graph.trace_dict[TraceType.TENSOR][name] = list(dict.fromkeys(trace_info))
                    trace_info = copy.deepcopy(converter_context.weights_trace_info[name])
                    if name in graph.trace_dict[TraceType.OP]:
                        trace_info.extend(graph.trace_dict[TraceType.OP][name])
                        graph.trace_dict[TraceType.OP][name] = list(dict.fromkeys(trace_info))
                    trace_info = copy.deepcopy(converter_context.weights_trace_info[name])
                    if name not in graph.trace_dict[TraceType.TENSOR] and \
                            name not in graph.trace_dict[TraceType.OP]:
                        trace_info.extend(graph.get_trace_info(node))
                        graph.set_trace_info(node, list(dict.fromkeys(trace_info)))

    # Static tensor in ONNX model will be translated to a constant op in ir graph
    # insert trace information for such scenario constant op
    def insert_constant_trace_info(self, framework_src_name, constant_node, converter_context):
        graph = converter_context.ir_graph
        if graph.enable_trace:
            static_trace_data = [(framework_src_name, TraceType.TENSOR)]
            graph.set_trace_info(constant_node, static_trace_data)
            buf = graph.get_output_buffers(constant_node)[0]
            graph.set_trace_info(buf, static_trace_data)
            # consider the const value could be calculated from exact op of framework model
            name = buf.name
            if name in converter_context.weights_trace_info:
                trace_info = copy.deepcopy(converter_context.weights_trace_info[name])
                if name in graph.trace_dict[TraceType.TENSOR]:
                    trace_info.extend(graph.trace_dict[TraceType.TENSOR][name])
                    graph.trace_dict[TraceType.TENSOR][name] = list(dict.fromkeys(trace_info))
                trace_info = copy.deepcopy(converter_context.weights_trace_info[name])
                if name in graph.trace_dict[TraceType.OP]:
                    trace_info.extend(graph.trace_dict[TraceType.OP][name])
                    graph.trace_dict[TraceType.OP][name] = list(dict.fromkeys(trace_info))
                trace_info = copy.deepcopy(converter_context.weights_trace_info[name])
                if name not in graph.trace_dict[TraceType.TENSOR] and \
                        name not in graph.trace_dict[TraceType.OP]:
                    trace_info.extend(graph.get_trace_info(constant_node))
                    graph.set_trace_info(constant_node, list(dict.fromkeys(trace_info)))

    # Add constant op to graph, if it doesn't exist
    def add_constant_src_op(self, input_name:str, const_op, converter_context):
        graph = converter_context.ir_graph
        if not graph.has_buffer(input_name) and converter_context.weights.has(input_name):
            input_constant_node = graph.add(const_op, [], [input_name])
            self.insert_constant_trace_info(input_name, input_constant_node, converter_context)
            graph.add_src_op_info(input_constant_node.op.name, [], [input_name])

    def get_supported_version(self, op_type):
        try:
            versions = []
            for schema_dict in self._op_schemas:
                # There may be more than one op schema associated with a given translation
                # This loop ensures that the right schema is selected
                if schema_dict.op_name == op_type:
                    versions = list(map(int, schema_dict.get_schemas().keys()))
            if not versions:
                raise RuntimeError("Could not find any op schema matching op type: {}.".format(op_type))

            return versions
        except Exception as e:
            raise RuntimeError("Could not retrieve supported versions for op: {}. The function returned with error: {}"
                               .format(op_type, str(e)))

    def register_op_schema(self, name, versions, unsupported_attrs=None):
        """
           Wraps Onnx's internal schema definition into a condensed op_schema_dict internal object (OpSchemaDict)
           which contains individual op_schema(s)(OpSchemaBase) that tie supported attributes,
           number of inputs and outputs to the appropriate op version

           :param name: The type of op to be registered
           :param versions : list of versions of the op to be registered. Note the versions must be available in
                             the Onnx spec.
           :param unsupported_attrs: A list of lists of unsupported attrs, which are in the Onnx spec
                                    for an op version but are not supported by the translation

           registers the resulting op_schema dictionary with the translation, as well as with a
           global schema registry

        """
        global ALL_SCHEMA_NAMES

        op_schema_idx = 0
        if unsupported_attrs:
            while len(unsupported_attrs) < len(versions):
                unsupported_attrs.append(unsupported_attrs[0])
        else:
            unsupported_attrs = [[] for _ in range(len(versions))]

        for i, version in enumerate(versions):
            if name not in ALL_SCHEMA_NAMES:
                log_warning("Unable to register converter supported Operation [{}:Version {}] with your Onnx installation. "
                            "Converter will bail if Model contains this Op.", name, version)
                # add a dummy op schema dict so functions can still be registered
                self._op_schemas.append(OpSchemaDict(name))
                continue
            # Note: get_schema uses version as maxInclusiveVersion and returns the schema
            # with the biggest version, which is not greater than specified version in
            # specified domain
            schema = defs.get_schema(name, version, '')
            op_schema = OpSchemaBase()
            op_schema.populate_op_schema(schema, unsupported_attrs[i])

            # if a schema dictionary already exists, then a new version is added. Otherwise,
            # a new op schema dictionary is created and the new schema is added
            schema_dicts = [schema_dict for schema_dict in self._op_schemas
                            if schema_dict.op_name == name]
            if schema_dicts:
                schema_dicts[0].add_schema(op_schema, version)
            else:
                op_schema_idx = len(self._op_schemas) - 1 if self._op_schemas else 0
                schema_dict = OpSchemaDict(name)
                schema_dict.add_schema(op_schema, version)
                self._op_schemas.append(schema_dict)

        OP_SCHEMA_REGISTRY[name.lower()] = self._op_schemas[op_schema_idx]
        return self._op_schemas[op_schema_idx]

    def op_schema(self, version: str = None, op_type: str = None):
        values = []

        # If version is provided, then all registered schemas matching that version are returned
        if version is not None:
            schema_versions = []
            for schema_dict in self._op_schemas:
                # There may be more than one op schema associated with a given translation
                # This loop ensures that the right schema is selected
                if op_type and not schema_dict.op_name == op_type:
                    continue
                schema_versions.append(schema_dict.get_schemas(version))

            if not schema_versions:
                raise RuntimeError("Could not find op schema for this translation matching version: {}, op type: {}"
                                   .format(str(version), op_type))

            return schema_versions
        else:
            # if no explicit version is requested, then we can retrieve the op_schemas associated
            # with this translation. If there is more than op schema dictionary registered then an
            # error is returned if no op type is requested. Otherwise, the latest schema is returned
            # for that op type. The latest is also returned if only one schema is registered.
            if not self._op_schemas:
                raise ValueError("No op schemas were registered for this translation: "
                                 "{}".format(self.__class__.__name__))
            elif len(self._op_schemas) == 1:
                values = list(self._op_schemas[0].get_schemas().values())
            elif len(self._op_schemas) > 1:
                if not op_type:
                    # internal error
                    raise AttributeError("Op type attribute must be provided when translation"
                                         " has more than one op schema registered")
                else:
                    schema_dicts = [schema_dict for schema_dict in self._op_schemas
                                    if schema_dict.op_name == op_type]
                    if not schema_dicts:
                        raise RuntimeError("Could not find any op schema matching op type: {}.".format(op_type))
                    values = list(schema_dicts[0].get_schemas().values())
            return values[-1]

    def _add_onnx_op(self, src_op, converter_context, model):
        '''
        Implementation of the addition of ONNX ops.
        Used by the add_op function for ops like if op, which contains subgraphs.
        It adds an ONNX op to the IrOpGraph in the converter_context.
        :param src_op: ONNX op to be added
        :param converter_context: Converter context object
        :returns: None
        '''
        src_type = converter_type(src_op.op_type, "onnx")
        try:
            # Check whether the op is a Block Op. It is a Block Op if and
            # only if domain is 'qti_aisw'
            if src_op.domain == "qti_aisw":
                src_type = converter_type(f'block_op_{src_op.op_type}', 'onnx')
                OnnxTranslations.apply_method_to_op(src_type,
                                                    OnnxTranslationBase.ADD_OP,
                                                    src_op,
                                                    converter_context)

            # check whether the op is a composable op or not
            # If so, then expand the composable operation and add individual nodes in the expansion.
            elif converter_context.composable_custom_op_collection and converter_context.composable_custom_op_collection.is_composable_op(src_op):
                self.__add_composable_op(src_op, converter_context, model)

            # check if layer is a registered custom op in an op collection.
            # If so, the layer is added and the outer loop continues.
            elif converter_context.custom_op_factory and src_op.op_type in converter_context.custom_op_factory.op_collection:
                src_type = converter_type('custom', "onnx")
                node = OnnxTranslations.apply_method_to_op(src_type,
                                                            OnnxTranslationBase.ADD_OP,
                                                            src_op,
                                                            converter_context)
                self.graph.add_src_op_info(node.op.name, [i for i in src_op.input], [o for o in src_op.output])


            elif src_op.domain in ['org.pytorch._caffe2']:
                src_type = converter_type(src_op.op_type, "onnx_caffe2")
                OnnxTranslations.apply_method_to_op(src_type,
                                                    OnnxTranslationBase.ADD_OP,
                                                    src_op,
                                                    converter_context)

            elif src_op.domain in ['spconv']:
                src_type = converter_type(src_op.op_type, "spconv")
                OnnxTranslations.apply_method_to_op(src_type,
                                                    OnnxTranslationBase.ADD_OP,
                                                    src_op,
                                                    converter_context)

            else:
                # If the op is not a custom operation, check the version and use the
                # native converter translation
                supported_version = OnnxTranslations.apply_method_to_op(src_type,
                                                                        OnnxTranslationBase.SUPPORTED_VERSION,
                                                                        src_op.op_type)
                op_info = OpVersionInfo()
                op_info.model_opset_version = converter_context.opset_version
                op_info.validate_op_ver(src_op, supported_version)

                OnnxTranslations.apply_method_to_op(src_type,
                                                    OnnxTranslationBase.ADD_OP,
                                                    src_op,
                                                    converter_context)

        except Exception as e:
            log_error(f"Failed to add op {src_op.op_type}")
            log_error("Node %s: %s" % (src_op.name, e))
            sys.exit(-1)

    def __add_composable_op(self, src_op, converter_context, model):
        """
        Expand the composable custom op node and add all the elementary nodes in the IR graph
        :param src_op: a Composable Custom op node
        :param model: a ONNX Model Proto
        :return:
        """
        expanded_nodes = ComposableCustomOp.expand(src_op, self.composable_custom_op_collection)

        # sub model is only required for Custom op. Sub Model will be created only if there is a programmable custom op
        # in the expansion
        sub_model = None
        custom_op_factory = converter_context.custom_op_factory
        for elem_op in expanded_nodes:
            # Check whether the op is a Block Op. It is a Block Op if and only
            # if domain is qti_aisw.
            if elem_op.domain == "qti_aisw":
                src_type = converter_type(f'block_op_{elem_op.op_type}', "onnx")
                OnnxTranslations.apply_method_to_op(src_type,
                                                     OnnxTranslationBase.ADD_OP,
                                                     elem_op,
                                                     converter_context)
            # check whether the op is a custom op or not
            elif custom_op_factory and elem_op.op_type in [operator.type_name for operator in custom_op_factory.custom_opdefs]:

                # create a ModelProto from the sub graph of the composable custom op
                if sub_model is None:
                    sub_model = ComposableCustomOp.create_model_from_function(src_op,
                                                                              expanded_nodes,
                                                                              converter_context.composable_custom_op_collection,
                                                                              model)

                if sub_model is None:
                    log_warning("Shape inference library should be provided for the programmable custom operations "
                                "of op type {} using --converter_op_package_lib option".format(elem_op.op_type))

                elem_op_type = converter_type('custom', "onnx")
                # dynamic flag should be true in this case since Custom onnx op for this node will not be present
                # in the custom op collection. We need to create a new custom onnx op from operator and src op.
                node = OnnxTranslations.apply_method_to_op(elem_op_type,
                                                            OnnxTranslationBase.ADD_OP,
                                                            elem_op,
                                                            converter_context,
                                                            dynamic=True,
                                                            model=sub_model)
                self.graph.add_src_op_info(node.op.name, [i for i in elem_op.input], [o for o in elem_op.output])
            elif elem_op.domain in ['org.pytorch._caffe2']:
                elem_op_type = converter_type(elem_op.op_type, "onnx_caffe2")
                OnnxTranslations.apply_method_to_op(elem_op_type,
                                                     OnnxTranslationBase.ADD_OP,
                                                     elem_op,
                                                     converter_context)
            else:
                elem_op_type = converter_type(elem_op.op_type, "onnx")
                supported_version = OnnxTranslations.apply_method_to_op(elem_op_type,
                                                                         OnnxTranslationBase.SUPPORTED_VERSION,
                                                                         elem_op.op_type)
                self.op_info.validate_op_ver(elem_op, supported_version)

                OnnxTranslations.apply_method_to_op(elem_op_type,
                                                     OnnxTranslationBase.ADD_OP,
                                                     elem_op,
                                                     converter_context)

    def _add_graph_initializers(self, branch, converter_context):
        for tensor in branch.initializer:
            np_tensor = get_initializer_value(tensor)
            # This condition is needed because the weights may have been added as part of main graph initializers
            if not converter_context.weights.has(tensor.name):
                was_scalar = False
                if not tensor.dims:
                    np_tensor = np_tensor.reshape(1,)
                    was_scalar = True
                converter_context.insert_weights(tensor.name, np_tensor, was_scalar)


class ElementwiseBinaryTranslationBase(OnnxTranslationBase, metaclass=ABCMeta):
    """
    Additional BaseClass for elementWiseBinary Ops(mul, prod, div and sub) since they need add_op to handle constant Op
    addition to graph
    """
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []
        self.operation = None
        self.numpy_op = None

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        params = extract_attributes(
            src_op,
            attr_infos=[('axis', 'i', -1),('broadcast', 'i', 0)],
            schema=self.op_schema(op_type=src_op.op_type),
            validate=True
        )
        op = self.extract_parameters(src_op, converter_context)
        # If op can be translated by NumPy, NonType will be returned
        # NonType op should not be added to graph
        if op is None:
            return
        output_names = self.extract_output_names(src_op, converter_context)

        if op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            self.add_src_op_info(op.name, src_op, graph)
            node = graph.add(op, [], output_names)
            self.insert_default_trace_info(src_op, node, converter_context)
            return node

        for input_name in self.input_names:
            const_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            # Add fetched constant op to graph, if it doesn't exist
            if const_op is not None:
                if not graph.has_buffer(input_name):
                    const_node = graph.add(const_op, [], input_name)
                    self.insert_constant_trace_info(input_name, const_node, converter_context)
                    graph.add_src_op_info(const_op.name, None, const_node.output_names[0])

        input_shapes = [graph.buffers[name].shape.dims for name in self.input_names]
        # According to ONNX doc for opset_v1 and opset_v6, only 2nd input can be broadcasted.
        # Thus, if broadcast is 1, we expect that the 1st input has max rank.
        if params['broadcast'] == 1 and 0 <= params['axis'] < (len(input_shapes[0]) - 1):
            # Calculate shape after broadcasting.
            broadcast_rank = len(input_shapes[0]) - len(input_shapes[1]) - params['axis']
            broadcast_shape = [1] * params['axis'] + input_shapes[1] + [1] * broadcast_rank

            # Get buffer after broadcasting.
            broadcast_input_name = self._get_broadcasted_buffer_name(
                src_op,
                self.input_names[1],
                broadcast_shape,
                graph
            )

            # Update 2nd input to broadcasted buffer.
            self.input_names[1] = broadcast_input_name

        # Add elementwise src op info
        self.add_src_op_info(op.name, src_op, graph)

        return graph.add_chained_eltwise_ops(op.operation, str(src_op.name), self.input_names, output_names[0])

    def extract_parameters(self, src_op, converter_context):
        self.input_names = self.extract_input_names(src_op, converter_context)
        const_input_ops = []
        for input_name in self.input_names:
            const_input_op = self.fetch_constant_op(input_name, converter_context, prunable=False, fail_if_dynamic=False)
            if const_input_op is not None:
                const_input_ops.append(const_input_op)
        if len(const_input_ops) == len(self.input_names):
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            out_dtype = converter_context.tensor_to_np_dtype.get(str(src_op.output[0]))
            const_input_data = [op.tensor for op in const_input_ops]
            numpy_data = self.numpy_op(*const_input_data)
            if out_dtype is None and numpy_data.dtype != bool:
                out_dtype = const_input_data[0].dtype
            data =  numpy_data.astype(out_dtype)
            was_scalar = all([converter_context.weights.was_scalar(input_name) for input_name in self.input_names])
            converter_context.insert_weights(str(src_op.output[0]), data, was_scalar, [src_op.name], self.input_names)
            return None

        return op_adapter.ElementwiseBinaryOp(str(src_op.name), operation=self.operation)

    def extract_input_names(self, src_op, converter_context):
        return [input_name for input_name in src_op.input]

    def _get_broadcasted_buffer_name(self, src_op, buffer_name, broadcast_shape, graph):
        # Check whether same Reshape op has already been inserted for broadcasting.
        # If yes, return the name of broadcasted buffer.
        for consumer in graph.buffers[buffer_name].consumers:
            if (
                isinstance(consumer.op, op_adapter.ReshapeOp)
                and list(consumer.op.shape) == broadcast_shape
            ):
                return consumer.output_names[0]
        # If no, insert a Reshape op after this buffer for broadcasting.
        broadcast_shape_str = "_".join(str(dim) for dim in broadcast_shape)
        broadcast_buffer_name = (
            self.input_names[1] + "_reshape_to_" + broadcast_shape_str
        )
        reshape_op = op_adapter.ReshapeOp(broadcast_buffer_name, shape=broadcast_shape)
        broadcast_axis_format = graph.buffers[self.input_names[0]].axis_format
        reshape_node = graph.add(
            reshape_op,
            [self.input_names[1]],
            [broadcast_buffer_name],
            axis_formats=[broadcast_axis_format],
        )
        # Add src op info for Reshape op.
        self.add_src_op_info(reshape_op.name, src_op, graph)
        # Set trace info for Reshape op.
        graph.set_trace_info(reshape_node, [(src_op.name, TraceType.OP)])
        graph.set_trace_info(
            graph.buffers[broadcast_buffer_name], [(src_op.name, TraceType.OP)]
        )
        return broadcast_buffer_name

class ElementwiseUnaryTranslationBase(OnnxTranslationBase, metaclass=ABCMeta):
    """
    Additional BaseClass for elementWiseUnary Op to extract parameters and handle constant Op
    addition to graph
    """
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.operation = None
        self.numpy_op = None

    def extract_parameters(self, src_op, converter_context):
        self.input_names = self.extract_input_names(src_op, converter_context)
        const_input_op = self.fetch_constant_op(str(src_op.input[0]), converter_context, prunable=False, fail_if_dynamic=False, fail_if_not_found=True)
        if const_input_op is not None:
            log_debug1(code_to_message.get_debugging_message("DEBUG_STATIC_OP")(src_op.name))
            out_dtype = converter_context.tensor_to_np_dtype.get(str(src_op.output[0]))
            data = self.numpy_op(const_input_op.tensor).astype(out_dtype)
            was_scalar = converter_context.weights.was_scalar(str(src_op.input[0]))
            converter_context.insert_weights(str(src_op.output[0]), data, was_scalar, [src_op.name], self.input_names)
            return None

        return op_adapter.ElementwiseUnaryOp(str(src_op.name), operation=self.operation)

    def extract_input_names(self, src_op, converter_context):
        return [input_name for input_name in src_op.input]

# -----------------------------------------------------------------
# Converter translations
# Note: ONNX doesn't have input op(s) but we create one for the IR
# -----------------------------------------------------------------
class OnnxInputTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)

    def extract_input_names(self, src_op, converter_context):
        raise NotImplementedError("extract_input_names() for {} not implemented ".format(str(self.__class__.__name__)))

    def extract_output_names(self, src_op, converter_context):
        raise NotImplementedError("extract_output_names() for {} not implemented ".format(str(self.__class__.__name__)))

    def add_op(self, src_op, converter_context, **kwargs):
        raise NotImplementedError("add_op() for {} not implemented. Call add_input_op() instead."
                                  .format(str(self.__class__.__name__)))

    def add_input_op(self, input_, graph, **kwargs):
        name = str(input_.name)

        if input_.type.tensor_type.elem_type == TensorProto.INT64 and graph.is_int64_input_preserved(name):
            input_dtype = np.dtype('int64')
        else:
            input_dtype = onnx_to_np_dtype[input_.type.tensor_type.elem_type]
        tensor_shape = input_.type.tensor_type.shape
        shape = [int(dim.dim_value) for dim in tensor_shape.dim]
        neg_idx = [idx for idx in range(len(shape)) if shape[idx] < 0]

        if neg_idx:
            raise RuntimeError('Negative/placeholder dimensions is not supported.'
                               'Expected shape: {} > 0\nNote: Dynamic input batch_size not supported. '
                               'Use --input_dim command to provide a static batch value'.format(shape))

        in_node = graph.add_input(name, shape, input_dtype=input_dtype)
        # add input node source information
        if graph.enable_trace:
            input_trace_data = [(name, TraceType.TENSOR)]
            graph.set_trace_info(in_node, input_trace_data)
            buf = graph.get_output_buffers(in_node)[0]
            graph.set_trace_info(buf, input_trace_data)
        return in_node


OnnxTranslations.register_translation(OnnxInputTranslation(),
                                      converter_type('input', 'onnx'),
                                      op_adapter.InputOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Dropout and other identity ops
# ------------------------------------------------------------------------------
class OnnxDropoutTranslation(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.register_op_schema('Dropout', [1, 6, 7, 10])

    def extract_parameters(self, src_op, converter_context):
        return op_adapter.IdentityOp(src_op.name)

    # Identity IR op expects an input number of 1, however, in the case of dropout,
    # we may get three inputs - data, ratio, training_mode.
    # So, override extract_input_names of OnnxIdentityTranslation to get only one input name.
    def extract_input_names(self, src_op, converter_context):
        return [str(src_op.input[0])]

    def extract_output_names(self, src_op, converter_context):
        return [str(src_op.output[0])]


OnnxTranslations.register_translation(OnnxDropoutTranslation(),
                                      converter_type('Dropout', 'onnx'),
                                      op_adapter.IdentityOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#   Class OpVersionInfo
# ------------------------------------------------------------------------------
# Returns name and version information about an op from a particular model
class OpVersionInfo:
    def __init__(self):
        self.model_opset_version = 0

    @staticmethod
    def update_schema_registry(src_op_type, op_version):
        """ Updates the schema registry so that get_op_schema(src_op_type) will always return the appropriate schema
            for the global model opset version """
        op_schema_dict = OP_SCHEMA_REGISTRY[src_op_type.lower()]
        op_schema_keys = list(op_schema_dict.get_schemas().keys())
        if op_schema_keys[-1] != str(op_version):
           op_schema_dict.reorder_op_schemas(str(op_version))

    def validate_op_ver(self, src_op, supported_version):
        """

        :param src_op: The op from the Onnx framework
        :param supported_version: The version of the op supported by the Onnx Converter
        :return: a warning if the opset_version for the source op does not match any version supported
                 by the converter
                 updates the schema registry if the src_op version is supported, so that any schema calls (self.op_schema()
                 or get_op_schema) will return the src_op_version.
        """

        # This uses the model version to extract the associated opset version for a given op
        # For example:
        # The scenarios are described below:
        # supported_version = [1, 6, 7]
        # Model_opset_version = 3,    Model_opset_version = 7,   Model_opset_version = 7,    Model_opset_version = 9
        # current_op_version = 1,     current_op_version = 7,    current_op_version = 1      current_op_version = 8
        #                                                        returns a warning for       returns a warning for
        #                                                        onnx installation support   converter support
        try:
            current_op_version = int(defs.C.get_schema(src_op.op_type, self.model_opset_version, '').since_version)
            if current_op_version not in supported_version:
                log_warning(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED")
                            (src_op.op_type, list(map(int, supported_version)), [current_op_version]))
            else:
                if self.model_opset_version != current_op_version and self.model_opset_version in supported_version:
                    log_warning(code_to_message.get_warning_message("WARNING_OP_VERSION_NOT_SUPPORTED_BY_ONNX")
                                (src_op.op_type, self.model_opset_version, current_op_version))
                self.update_schema_registry(src_op.op_type, current_op_version)
        except RuntimeError as e:
            # Throw an error here since model contains an op or a max_version that is not part of the current onnx
            # installation.
            # Note: re-raising error since the onnx error message is not very informative
            raise RuntimeError("Your Onnx installation does not support Operation [{}:Version {}] found in model. "
                               "Got: {}".format(src_op.op_type, self.model_opset_version, str(e)))

    def set_global_op_ver(self, model):
        """ Sets the global op version supported by the model"""
        # Get the global opset version
        if len(model.opset_import) > 1:
            log_warning(code_to_message.get_warning_message("WARNING_OPSET_VERSION"))

        for opset in model.opset_import:
            # Setting model opset version to model onnx version/ai.onnx version.
            if opset.domain == "" or opset.domain == "ai.onnx":
                self.model_opset_version = opset.version