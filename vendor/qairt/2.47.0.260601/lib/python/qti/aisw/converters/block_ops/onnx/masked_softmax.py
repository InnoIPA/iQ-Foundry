# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""ONNX MaskedSoftmax BlockOperator code"""

import onnx
from onnxscript import script
from onnxscript.values import OnnxFunction

from qti.aisw.converters.block_ops.onnx.onnx_block_op_base import (
    QnnOnnxBlockOp,
    qcom_block_op_domain,
)
from qti.aisw.converters.common.converter_ir.op_properties.masked_softmax import (
    MODE_UNCOMPRESSED_VAL,
    MODE_COMPRESSED_VAL,
    MODE_DEFAULT_VAL,
)


class MaskedSoftmax(QnnOnnxBlockOp):
    """ONNX MaskedSoftmax Block Operator

    This class represents a MaskedSoftmax Block Operator in ONNX. The operator
    is defined in onnxscript, so both getOnnxScriptFunc and getOnnxFuncProto
    are implemented.

    :ivar min_opset: Initialized to 15.
    :ivar max_opset: Initialized to 17.

    :ivar name: Initialized to 'MaskedSoftmax'.
    :ivar mode: Mode to materialize the implementation in. Available options are
        MODE_UNCOMPRESSED_VAL, or MODE_COMPRESSED_VAL. The default value is
        specified in MODE_DEFAULT_VAL.
    """

    def __init__(
        self,
        onnx_opset_version: int,
        mode: int = MODE_DEFAULT_VAL,
        aisw_opset_version: int = 1,
    ):
        self.name = "MaskedSoftmax"
        self.max_opset = 17
        self.min_opset = 15
        self.mode = mode
        if aisw_opset_version != 1:
            raise ValueError(
                f"Operator {self.name} is only supported in "
                f"version 1 of domain {qcom_block_op_domain()}."
            )
        super(MaskedSoftmax, self).__init__(onnx_opset_version, aisw_opset_version)

    def getOnnxScriptFunc(self) -> OnnxFunction:
        """Creates an onnxscript function representation. Implementation depends
           on `self.mode`.

        :raises ValueError: If self.mode is not one of MODE_UNCOMPRESSED_VAL or
            MODE_COMPRESSED_VAL.
        :return: onnxscript function
        :rtype: onnxscript.values.OnnxFunction

        """
        opset = self.opset
        qti_aisw = self.aisw_opset

        if self.mode == MODE_UNCOMPRESSED_VAL:

            @script(qti_aisw, opset)
            def MaskedSoftmax(data, mask, mode: int = MODE_UNCOMPRESSED_VAL):
                """Uncompressed MaskedSoftmax operator in ONNX script

                :param data: Data to mask.
                :type data: Tensor of shape [batch, height, width,
                    channel]. batch == 1.
                :param mask: Mask to apply to data. Values to be kept should
                    have a 0 in the mask tensor, values to be masked off should
                    be a large negative number (e.g. -10000). Mask is broadcast
                    to data dimensions.
                :type mask: Tensor of shape [batch, channel]. batch == 1.
                """
                return opset.Softmax(data + mask)
        elif self.mode == MODE_COMPRESSED_VAL:

            @script(qti_aisw, opset)
            def MaskedSoftmax(data, mask, mode: int = MODE_COMPRESSED_VAL):
                """Compressed MaskedSoftmax operator in ONNX script

                :param data: Concatenated data sequences to mask.
                :type data: Tensor of shape [batch,
                    height, width, channel]. batch == 1 and width == channel.
                :param mask: Compressed mask to apply to data. Each element
                    represents a sequence length. Sum of the elements must be
                    less than or equal to the channel.
                :type mask: Tensor of shape [batch, number of concatenated
                    sequences]. batch == 1.
                """

                dataShape = opset.Shape(data)
                lastIndex = opset.Constant(value_int=-1)
                maskSize = opset.Gather(dataShape, lastIndex)

                # Expands the mask into an uncompressed version
                zeroInt64 = opset.Constant(value_int=0)
                oneInt64 = opset.Constant(value_int=1)
                maxSeqLenRange = opset.Range(zeroInt64, maskSize, oneInt64)
                transposedMask = opset.Transpose(mask)
                cumSumMask = opset.CumSum(transposedMask, zeroInt64)
                cumSumMaskExcl = opset.CumSum(transposedMask, zeroInt64, exclusive=1)
                unsqueezeCumSum = opset.Unsqueeze(cumSumMask, zeroInt64)
                unsqueezeCumSumExcl = opset.Unsqueeze(cumSumMaskExcl, zeroInt64)
                transposeCumSum = opset.Transpose(unsqueezeCumSum)
                transposeCumSumExcl = opset.Transpose(unsqueezeCumSumExcl)
                rangeLessThanMask = opset.Less(maxSeqLenRange, transposeCumSum)
                rangeLessThanMaskExcl = opset.Less(maxSeqLenRange, transposeCumSumExcl)
                notRangeLessThanMaskExcl = opset.Not(rangeLessThanMaskExcl)
                mask = opset.And(notRangeLessThanMaskExcl, rangeLessThanMask)
                floatMask = opset.Cast(mask, to=1)
                floatMaskT = opset.Transpose(floatMask, perm=[0, 2, 1])
                expandedMaskFloat = opset.MatMul(floatMaskT, floatMask)
                expandedMask = opset.Cast(expandedMaskFloat, to=9)

                # Performs the actual masking and softmax
                zerof = opset.Constant(value_float=0.0)
                minus10000f = opset.Constant(value_float=-10000.0)
                softmaxIn = opset.Where(expandedMask, data, minus10000f)
                softmaxOut = opset.Softmax(softmaxIn)
                rv = opset.Where(expandedMask, softmaxOut, zerof)
                return rv
        else:
            raise ValueError(
                f"{self.name} contains an unrecognized mode value "
                f"of {self.mode}. Accepted values are "
                f"MODE_UNCOMPRESSED_VAL ({MODE_UNCOMPRESSED_VAL}) "
                f"or MODE_COMPRESSED_VAL ({MODE_COMPRESSED_VAL})."
            )

        return MaskedSoftmax

    def getOnnxFuncProto(self) -> onnx.FunctionProto:
        return self.getOnnxScriptFunc().to_function_proto()
