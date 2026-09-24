# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from itertools import groupby

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import (
    ImageRepresentation,
    PostProcessor,
)


class LPRNETPostProcessor(PostProcessor):
    """Used for LPRNET license plate prediction.

    Attributes:
        vocab (dict[int, str]): Mapping of class indices to their
                    corresponding license plate characters or other information.
        class_axis (int): Axis along which the model output is expected.
                     Defaults to -1 (last axis).

    Note: The vocabulary includes numbers 0-9, and a range of province codes in China.
          Additionally, it covers letters A-Z, an underscore character,
          and a special code for 'police'. Class index -1 corresponds to an empty string.
    """

    vocab: dict[int, str] = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
            9: '9', 10: '<Anhui>', 11: '<Beijing>', 12: '<Chongqing>', 13: '<Fujian>',
            14: '<Gansu>', 15: '<Guangdong>', 16: '<Guangxi>', 17: '<Guizhou>', 18: '<Hainan>',
            19: '<Hebei>', 20: '<Heilongjiang>', 21: '<Henan>', 22: '<HongKong>', 23: '<Hubei>',
            24: '<Hunan>', 25: '<InnerMongolia>', 26: '<Jiangsu>', 27: '<Jiangxi>', 28: '<Jilin>',
            29: '<Liaoning>', 30: '<Macau>', 31: '<Ningxia>', 32: '<Qinghai>', 33: '<Shaanxi>',
            34: '<Shandong>', 35: '<Shanghai>', 36: '<Shanxi>', 37: '<Sichuan>', 38: '<Tianjin>',
            39: '<Tibet>', 40: '<Xinjiang>', 41: '<Yunnan>', 42: '<Zhejiang>', 43: '<police>',
            44: 'A', 45: 'B', 46: 'C', 47: 'D', 48: 'E', 49: 'F', 50: 'G', 51: 'H', 52: 'I',
            53: 'J', 54: 'K', 55: 'L', 56: 'M', 57: 'N', 58: 'O', 59: 'P', 60: 'Q', 61: 'R',
            62: 'S', 63: 'T', 64: 'U', 65: 'V', 66: 'W', 67: 'X', 68: 'Y', 69: 'Z', 70: '_', -1: ''
        }  # yapf: disable

    def __init__(self, class_axis: int = -1):
        """Initializes an instance of the LPRNETPostProcessor class.

        Args:
            class_axis (int): Axis along which the model output is expected. Defaults to 1.
        """
        self.class_axis = class_axis

    def execute(self, input_sample: ImageRepresentation) -> ImageRepresentation:
        """Processes LPRNET output and predicts the License Plate number.

        Args:
            input_sample (ImageRepresentation): Input image representation object

        Returns:
            ImageRepresentation: Processed image representation object
        """
        data = input_sample.data[0].squeeze()  # shape: [88, 71]
        postprocessed_data = self.postprocess(data)
        # Replace original data with updated list
        input_sample.data = [postprocessed_data]
        return input_sample

    def postprocess(self, logits_vector: np.ndarray) -> str:
        """Applies postprocessing to a single image model output.

        Args:
            logits_vector (np.ndarray): Logits matrix from the LPRNET model output

        Returns:
            str: Predicted License Plate number
        """
        # logits_vector: single image model output
        # logits_vector is logits matrix, with num_classes=71 (as in vocab)
        predicted_classes = np.argmax(logits_vector, axis=self.class_axis)
        # do greedy detection: ignore consecutive repetitions and ignore class 70:'_'
        unique_predicted_classes = [x[0] for x in groupby(predicted_classes) if x[0] != 70]
        processed_output = "".join([self.vocab[x] for x in unique_predicted_classes])
        return processed_output
