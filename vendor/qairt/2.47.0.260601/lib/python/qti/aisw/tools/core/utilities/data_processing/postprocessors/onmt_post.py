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
# SOURCE: https://github.com/NVIDIA/FasterTransformer/tree/54e1b4a981f00b83bc22b2939743ec1e58164b86
# LICENSE: https://github.com/NVIDIA/FasterTransformer/blob/54e1b4a981f00b83bc22b2939743ec1e58164b86/LICENSE
###############################################################################
# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from typing import Optional

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import (
    PostProcessor,
    TextRepresentation,
)
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class OpenNMTPostprocessor(PostProcessor):
    """Postprocessor for OpenNMT Model with the WMT20 test dataset."""

    def __init__(
        self,
        sentencepiece_model_path: str | os.PathLike,
        unrolled_count: Optional[int] = 26,
    ):
        """Initializes the OnmtPostProcessor.

        Args:
            sentencepiece_model_path (str | os.PathLike): The path to the SentencePiece model.
            unrolled_count (Optional[int]): The count for unfolding. Defaults to 26.
        """
        self.unrolled_count = unrolled_count
        self.sentencepiece_model_path = sentencepiece_model_path
        self.validate()

    def validate(self):
        """Validates the OpenNMT Postprocessor parameters."""
        if not os.path.exists(self.sentencepiece_model_path):
            raise ValueError(
                f"Sentencepiece model does not exist at the path {self.sentencepiece_model_path}"
            )
        if not isinstance(self.unrolled_count, int) or self.unrolled_count < 1:
            raise ValueError("Unrolled count must be a positive integer")
        try:
            sentencepiece = Helper.safe_import_package("sentencepiece", "0.2.0")
            self.sp = sentencepiece.SentencePieceProcessor(self.sentencepiece_model_path)
        except Exception as e:
            raise ValueError(
                f"Failed to Create SentencePieceProcessor with the given  \
                    sentencepiece_model_path={self.sentencepiece_model_path}. Reason: {e}"
            )

    def execute(self, input_sample: TextRepresentation) -> TextRepresentation:
        """Execute OpenNMT postprocessing for the given text sample.

        Args:
            input_sample (TextRepresentation): The input text representation.

        Returns:
            TextRepresentation: The output text representation.
        """
        # Get the beams output from the input sample
        beams = input_sample.data[0]
        # Reshape the beams to a single row
        beams = beams.reshape([1, -1])
        # Get the fields and dataset metadata
        fields = input_sample.metadata["fields"]
        # Build the target tokens using OpenNMT postprocessor
        predicted_sentences = OpenNMTPostprocessor.build_target_tokens(beams[0], fields)
        # Decode the predicted sentences
        out = self.sp.decode(predicted_sentences[: self.unrolled_count])
        # Update the input sample data with the decoded output
        input_sample.data = np.array([out])
        return input_sample

    @classmethod
    def build_target_tokens(
        cls,
        predicted_output_sequence: list[int],
        fields: dict[str, "Field"],  # noqa: F821
    ) -> list[str]:
        """Builds target tokens from predicted output and source vocabulary.

        Args:
            predicted_output_sequence: Predicted output sequence of indices.
            fields: Dictionary containing the field objects for the prediction task.

        Returns:
            A list of target tokens corresponding to the predicted output sequence.

        Note:
            This function assumes that the predicted output sequence contains indices
            from both the source and target vocabularies.
            It uses this information to construct a list of target tokens, including
            EOS token if present.
        """
        target_field = dict(fields)["tgt"].base_field
        target_vocab = target_field.vocab
        tokens = []

        for tok in predicted_output_sequence:
            if tok < len(target_vocab):
                tokens.append(target_vocab.itos[tok])
            if tokens[-1] == target_field.eos_token:
                tokens = tokens[:-1]
                break
        return tokens
