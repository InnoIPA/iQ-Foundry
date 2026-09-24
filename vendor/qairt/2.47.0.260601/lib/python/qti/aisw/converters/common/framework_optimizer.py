# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC, abstractmethod


class FrameworkOptimizer(ABC):
    def __init__(self):
        """
        Initialize the FrameworkOptimizer object.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Return string representation of the optimization. It should return the
        name of the optimization. This will be helpful when registering all the
        optimizations and running it from the OptimizerManager kind of class
        which doesn't know about individual optimizations.

        e.g. "ONNX - Constant Folding Optimizer"
        """
        pass

    @abstractmethod
    def optimize(self, **kwargs):
        """
        Method to apply specific graph optimization.
        """
        pass
