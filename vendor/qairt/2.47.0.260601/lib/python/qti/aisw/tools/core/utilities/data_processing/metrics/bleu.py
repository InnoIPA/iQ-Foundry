##############################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SOURCE: https://github.com/mjpost/sacrebleu
# LICENCE: https://github.com/mjpost/sacrebleu/blob/master/LICENSE.txt
###############################################################################
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.tools.core.utilities.data_processing.metrics import Metric
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class BLEUMetric(Metric):
    """Class for BLEU metric using the sacrebleu library.

    Args:
        decimal_places (int): The number of decimal places to round the BLEU score to. Default is 2.
    """

    def __init__(self, decimal_places: int = 2):
        """Initializes a new instance of the `BLEUMetric` class."""
        # Note: super init is called to setup metric_state
        super().__init__()
        self.decimal_places = decimal_places
        self._validate()

    def _validate(self):
        """Validate the input parameters."""
        if not (isinstance(self.decimal_places, int) and self.decimal_places > 0):
            raise ValueError("decimal_places must be a positive integer")
        sacrebleu = Helper.safe_import_package("sacrebleu", "2.3.1")
        if sacrebleu is None:
            raise RuntimeError(
                "The sacrebleu module was not found."
                " Please install the sacrebleu library to use the BLEU Metric."
            )

    def finalize(self) -> dict[str, float]:
        """Calculate and return the BLEU score.

        Returns:
            dict: A dictionary containing the BLEU score.
        """
        sacrebleu = Helper.safe_import_package("sacrebleu", "2.3.1")  # noqa: F841
        from sacrebleu.metrics import BLEU

        count = len(self.metric_state)
        save_results = {}
        if count > 0:
            annotations_file_path = self.metric_state[0].annotation.data
            ref_labels = [line.strip() for line in open(annotations_file_path).readlines()]
            predicted_tokens = [None] * count
            for input_data in self.metric_state:
                idx = input_data.idx
                predicted_tokens[idx] = input_data.data[0].item()
            bleu = BLEU()
            score = bleu.corpus_score(predicted_tokens, [ref_labels]).score
            save_results["bleu"] = round(score, self.decimal_places)
        return save_results
