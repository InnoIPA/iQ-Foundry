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

import argparse
import os
import pickle

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import NDArrayRepresentation, PreProcessor
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class OpenNMTPreprocessor(PreProcessor):
    """A pre-processor for OpenNMT that reads text data and applies processing.

    Attributes:
        vocab_path (os.PathLike): The path to the vocabulary file.
        src_seq_len (int, optional): The source sequence length. Defaults to 128.
    """

    def __init__(
        self,
        vocab_path: os.PathLike,
        src_seq_len: int = 128,
    ):
        """Initializes the OpenNMTPreprocessor instance.

        Args:
            vocab_path (os.PathLike): The path to the vocabulary file.
            src_seq_len (int, optional): The source sequence length. Defaults to 128.
        """
        self.vocab_path = vocab_path
        self.src_seq_len = src_seq_len
        # Validate the input arguments
        self.validate()
        # Setup the vocabulary
        self.setup_vocab()

    def setup_vocab(self) -> None:
        """Loads the vocabulary fields from file."""
        # Load the vocabulary fields from file
        try:
            with open(self.vocab_path, "rb") as f:
                self.fields = pickle.load(f)
            # Remove the 'corpus_id' field from the loaded vocabulary
            self.fields.pop("corpus_id")
        except Exception:
            raise RuntimeError(f"Error loading vocabulary from file: {self.vocab_path}")

    def validate(self):
        """Validates the input arguments.

        Raises:
            ValueError: If the vocab_path is invalid
            ImportError: If OpenNMT-py library is not installed
        """
        # Check if the vocab_path is valid
        if not (isinstance(self.vocab_path, os.PathLike) or os.path.exists(self.vocab_path)):
            raise ValueError("Invalid vocab path: {}".format(self.vocab_path))

        onmt = Helper.safe_import_package("onmt", "2.3.0")
        if onmt is None:
            raise ImportError(
                "The OpenNMT-py library was not found. Please install the OpenNMT-py"
                " library to use the OpenNMTPostprocessor."
            )

    def execute(self, input_sample: NDArrayRepresentation):
        """Executes the pre-processing logic for the given data.

        Args:
            input_sample (TextRepresentation): The text representation of the input data.

        Returns:
            TextRepresentation: The processed text representation.

        Raises:
            RuntimeError: If an error occurs during processing.
        """
        onmt = Helper.safe_import_package("onmt", "2.3.0")  # noqa: F841
        import onmt.inputters as inputters

        if len(input_sample.data) != 1:
            raise ValueError(
                "OpenNMTPreprocessor expects input_sample to contain a"
                " single string numpy array in data field"
            )
        data = np.array(input_sample.data)
        # Configure the input reader for text data
        opt = argparse.Namespace(data_type="text")
        src_reader = inputters.str2reader["text"].from_opt(opt)
        # Create a data structure for the processed data
        src_data = {"reader": src_reader, "data": data, "dir": None}
        # Configure the dataset using the provided fields and readers
        _readers, _data = inputters.Dataset.config([("src", src_data)])

        # Create an ordered iterator over the dataset with batch size 1
        dataset = inputters.Dataset(
            self.fields, readers=_readers, data=_data, sort_key=inputters.str2sortkey["text"]
        )
        data_iter = inputters.OrderedIterator(dataset=dataset, batch_size=1)

        # Convert the iterator to a list of processed examples
        data_list = list(data_iter)

        # Set metadata for the input sample
        input_sample.metadata["fields"] = self.fields
        input_sample.metadata["dataset"] = dataset

        # Initialize an empty list to store the output files
        out_data = []

        # Iterate over each processed example in the list
        for i in range(len(data_list)):
            # Get the source text and lengths from the current example
            src = data_list[i].src[0]
            src_lengths = data_list[i].src[1]

            # Pad the source text to match the desired sequence length
            pad_length = self.src_seq_len - src.shape[0]

            src = np.concatenate([src, np.zeros((pad_length, src.shape[1], 1)).astype(np.int64)], axis=0)

            # Append the padded source text and lengths to the output files list
            out_data.append(src)
            out_data.append(src_lengths.numpy())

        # Set the output data for the input sample
        input_sample.data = out_data
        # Return the processed input sample
        return input_sample
