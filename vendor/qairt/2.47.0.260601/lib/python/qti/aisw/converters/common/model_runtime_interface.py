# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Text, Tuple

import numpy as np
from qti.aisw.converters.common.utils.framework_utils import TensorInfo


class IModelRuntime(ABC):
    def __init__(self) -> None:
        """
        Initializing runtime object.
        """

    @abstractmethod
    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets input tensor information.

        :return Dict[Text, TensorInfo]: Input names to tensor information.
        """
        pass

    @abstractmethod
    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Gets output tensor information

        :return Dict[Text, TensorInfo]: Output names to tensor information.
        """
        pass

    @abstractmethod
    def execute_inference(
        self, inputs: Dict[Text, np.ndarray], output_names: Optional[List] = []
    ) -> Tuple[bool, Dict[Text, np.ndarray]]:
        """
        Run the inference of given model.

        :param Dict[Text, np.ndarray] input_data: Dict containing input tensor name
            to corresponding tensor data.
        :param Optional[List] output_names:Optional list of output names for
            inference.
        :return Tuple[bool, Dict[Text, np.ndarray]]: Tuple of two values. 1st value
            represents the status of inference and 2nd value represents the Dict
            containing output tensor name as key and its computed numpy array output as value.
        """
        pass
