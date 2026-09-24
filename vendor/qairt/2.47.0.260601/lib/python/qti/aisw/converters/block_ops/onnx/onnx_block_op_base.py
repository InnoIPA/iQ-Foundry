# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import onnx
import onnxscript
from onnxscript.values import Opset
import abc


def qcom_block_op_domain() -> str:
    """Returns the BlockOps domain as a string"""
    return "qti_aisw"


def _get_onnx_opset_version(model: onnx.ModelProto) -> int:
    """Return the ONNX opset version from a model's opset imports.

    :param model: The model to inspect.
    :type model: onnx.ModelProto
    :return: The ONNX opset version.
    :rtype: int
    :raises ValueError: If no ONNX opset entry is found in the model.
    """
    for opset in model.opset_import:
        if opset.domain == "":  # empty string is the ONNX domain
            return opset.version
    raise ValueError("ONNX opset version could not be found in model")


class QnnOnnxBlockOp(metaclass=abc.ABCMeta):
    """Abstract class to materialize implementations of a single Block Op
       in ONNX.

    The following instance variables are set by the base class constructor:
    :ivar onnx_opset_version: The ONNX opset version as an integer that this
        instance will materialize a Block Op implementation in.
    :vartype onnx_opset_version: int
    :ivar aisw_opset_version: The AISW opset version as an integer that this
        instance will materialize a Block Op implementation in.
    :vartype aisw_opset_version: int

    The following instance variables *are not* set by the base class
    constructor, and the derived class constructor *must* set them before
    calling the base class constructor:
    :ivar min_opset: Integer representing the minimum ONNX opset version that
        this operator is supported in. This is used to throw a nice error if the
        user requests an implementation in an unsupported version of ONNX.
    :vartype min_opset: int
    :ivar max_opset: Integer representing the maximum ONNX opset version that
        this operator is supported in. This is used to throw a nice error if the
        user requests an implementation in an unsupported version of ONNX.
    :ivar name: The name of the Block Op.
    :vartype name: str

    """

    def __init__(self, onnx_opset_version: int, aisw_opset_version: int = 1):
        """Constructs a QnnOnnxBlockOp object, from which a concrete
           implementation of the Block Op can be created. This base class
           constructor is intended to be called *at the end* of the derived
           class constructor.

        :param onnx_opset_version: Sets value of onnx_opset_version.
        :param aisw_opset_version: Sets value of aisw_opset_version.
            Defaults to 1.

        """
        if onnx_opset_version < self.min_opset or onnx_opset_version > self.max_opset:
            raise ValueError(
                f"BlockOp {self.name} only supports ONNX opset "
                f"versions between {self.min_opset} and "
                f"{self.max_opset}. Please choose another "
                "ONNX opset version."
            )
        self.opset = Opset("", onnx_opset_version)
        self.aisw_opset = Opset(qcom_block_op_domain(), aisw_opset_version)

    def getOnnxScriptFunc(self) -> onnxscript.values.OnnxFunction:
        """Creates an onnxscript function implementation of this Block Op.  Not
           guaranteed to be implemented. Block Ops that are incapable of
           producing an ONNX script representation need not override this
           function.

        :return: An onnxscript representation of the Block Op.
        :rtype: onnxscript.values.OnnxFunction
        :raises ValueError: If not implemented by deriving class.
        """
        raise ValueError(
            f"BlockOp {self.name} does not have an ONNX script "
            "representation. Please use getOnnxFuncProto instead "
            "to get a raw function proto."
        )

    def getOnnxFuncProto(self) -> onnx.FunctionProto:
        """Creates an ONNX function proto implementation of the Block Op. All
           Block Ops MUST be able to return a function proto, regardless of
           their ability to return an ONNXScript function.

        :return: An ONNX function proto representation of the Block Op.
        :rtype: onnx.FunctionProto
        :raises NotImplementedError: If not implemented by deriving class.
        """
        raise NotImplementedError
