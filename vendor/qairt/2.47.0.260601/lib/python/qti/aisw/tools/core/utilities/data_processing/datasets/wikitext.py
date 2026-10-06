# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from typing import Iterable, Literal, Optional

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import Annotation, NDArrayRepresentation
from qti.aisw.tools.core.utilities.data_processing.datasets import IndexableDataset
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class WikiTextDataset(IndexableDataset):
    """Tokenize data from files using GPT2TokenizerFast or bloom tokenizer."""

    TOKENIZER_INFO = """tokenizer_model_name_or_path param can be any one of the below:
        1) A string, the model id of a predefined tokenizer hosted inside a model repo on huggingface.co.
        2) A string, the model id of a predefined tokenizer from huggingface.co (user-uploaded)
             and cache (e.g. "deepset/roberta-base-squad2")
        3) A path to a directory containing vocabulary files required by the tokenizer,
                 for instance saved using the save_pretrained() method, e.g., ./my_model_directory/.
        """

    def __init__(
        self,
        tokenizer_model_name_or_path: str | os.PathLike,
        inputlist_path: os.PathLike | str,
        sequence_length: int,
        past_shape: Optional[list[int]] = None,
        calibration_indices: Optional[list[int] | os.PathLike] = None,
        max_samples: Optional[int] = None,
        use_calibration: Optional[bool] = False,
        past_sequence_length: int = 1,
        num_past: int = 0,
        position_id_required: bool = True,
        mask_dtype: Literal["int64", "float32"] = "float32",
    ):
        """Initialize WikiText2Dataset dataset.

        Args:
            tokenizer_model_name_or_path (str | os.PathLike): tokenizer_model_name_or_path param can be
            any one of the below:
            1) A string, the model id of a predefined tokenizer hosted inside a model repo on huggingface.co.
            2) A string, the model id of a predefined tokenizer from huggingface.co (user-uploaded)
             and cache (e.g. "deepset/roberta-base-squad2")
            3) A path to a directory containing vocabulary files required by the tokenizer,
                 for instance saved using the save_pretrained() method, e.g., ./my_model_directory/.
            inputlist_path (os.PathLike | str): Path to the file containing text files.
            sequence_length (int): Length of each sequence.
            past_shape (List[int]): Shape of past sequences.
            calibration_indices (Optional[List[int]], optional): Calibration indices. Defaults to None.
            max_samples (Optional[int], optional): Maximum number of samples. Defaults to None.
            use_calibration (bool, optional): Whether to use calibration data. Defaults to False.
            past_sequence_length (int, optional): Length of past sequences. Defaults to 0.
            num_past (int, optional): Number of past sequences. Defaults to 0.
            position_id_required (bool, optional): Whether position IDs are required. Defaults to True.
            mask_dtype (Literal["int64", "float32"], optional): Data type for masks. Defaults to "float32".
        """
        self.tokenizer_model_name_or_path = tokenizer_model_name_or_path
        self.inputlist_path = inputlist_path
        self.sequence_length = sequence_length
        self.calibration_indices = calibration_indices
        self.max_samples = max_samples
        self.use_calibration = use_calibration
        self.past_shape = past_shape
        self.past_sequence_length = past_sequence_length
        self.num_past = num_past
        self.position_id_required = position_id_required
        self.mask_dtype = mask_dtype
        if isinstance(self.calibration_indices, (str, os.PathLike)):
            calibration_indices_str = open(self.calibration_indices).read().strip()
            self.calibration_indices = [int(idx) for idx in calibration_indices_str.split(",")]
        self.validate()
        self._setup()

    def validate(self):
        """Validate the  parameters provided to WikitText2Tokenized dataset."""
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        if transformers is None:
            raise ImportError("transformers library is required to use WikitText2Tokenized dataset")

        if self.inputlist_path and not os.path.exists(self.inputlist_path):
            raise FileNotFoundError(
                f"Inputlist file provided does not exist. provided: {self.inputlist_path}"
            )

        if self.max_samples is not None:
            if type(self.max_samples) is not int:
                raise ValueError("max_samples must be a integer")
            if self.max_samples == 0:
                raise ValueError("max_samples cannot be 0. Provide valid values")

        if not isinstance(self.use_calibration, bool):
            raise ValueError("use_calibration must be a boolean")
        if self.use_calibration and not self.calibration_indices:
            raise ValueError("When use_calibration is set to True, user must provide calibration_indices")

        if self.mask_dtype not in ["int64", "float32"]:
            raise ValueError(f"Mask dtypes must be one of 'int64' or 'float32'. Provided: {self.mask_dtype}")

        if not isinstance(self.position_id_required, bool):
            raise ValueError(f"Position id required must be a boolean. Provided: {self.position_id_required}")

        if type(self.sequence_length) is not int or self.sequence_length < 0:
            raise ValueError("Sequence length must be an non negative number")

        if type(self.past_sequence_length) is not int or self.past_sequence_length < 0:
            raise ValueError("Past Sequence length must num_past be a positive number")

        if type(self.num_past) is not int or self.num_past < 0:
            raise ValueError("num past must be an non negative integer")

        if self.past_shape is not None and not (
            isinstance(self.past_shape, Iterable) and all(isinstance(dim, int) for dim in self.past_shape)
        ):
            raise ValueError("Past shape must be a list of intergers")

        if (
            self.calibration_indices
            and isinstance(self.calibration_indices, Iterable)
            and not any((isinstance(idx, int) and idx > 0) for idx in self.calibration_indices)
        ):
            raise ValueError(
                f"The calibration indices must be postive integers. Provided: {self.calibration_indices}"
            )

    def _setup(self):
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        # read the text and tokenize
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_model_name_or_path)
        except Exception:
            raise RuntimeError(f"Failed to setup the tokenizer. {self.TOKENIZER_INFO}")
        self.data = self.prepare_data()
        self.annotation = self.prepare_annotations()

    def prepare_data(self) -> list[list[np.ndarray]]:
        """Prepare data for the dataset.

        Returns:
            List[List[np.ndarray]]: Processed (Tokenized) data.
        """
        tokens = self.tokenize()
        input_ids_list = tokens["input_ids"]
        attention_mask_list = tokens["attention_mask"]
        pos_ids = np.arange(self.sequence_length, dtype=np.int64)
        if self.num_past:
            zero_past = np.zeros(self.past_shape, dtype=np.float32)
        input_data = []
        for i, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            input_ids = np.array(input_ids, dtype=np.int64)
            attention_mask = np.array(attention_mask, dtype=self.mask_dtype)
            if self.position_id_required:
                data = [input_ids, pos_ids, attention_mask]
            else:
                data = [input_ids, attention_mask]

            for past_idx in range(self.num_past):
                data.append(zero_past)
            input_data.append(data)
        if self.max_samples not in [None, -1]:
            input_data = input_data[: self.max_samples]
        return input_data

    def prepare_annotations(self) -> list[Annotation]:
        """Prepare annotations for the dataset.

        Returns:
            List[Annotation]: Annotations.
        """
        annotations = []
        for data in self.data:
            annotations.append(Annotation(data=data[0]))
        return annotations

    def __getitem__(self, index) -> NDArrayRepresentation:
        """Returns a single data point at the specified index.

        Args:
            index (int): The index of the data point to return.

        Returns:
            NDArrayRepresentation: An object containing the data, annotation, and
             metadata for the requested data point.
        """
        metadata = {"tokenizer": self.tokenizer}
        return NDArrayRepresentation(
            data=self.data[index], annotation=self.annotation[index],
             metadata=metadata, idx=index
        )

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.data)

    def tokenize(self) -> dict:
        """Read the text from files and tokenize.

        Returns a dictionary where each key is a token type and the value is a list of chunks
        of length `self.sequence_length`. Each chunk is a list of integers representing tokens.

        Returns a dictionary {
            'input_ids': [[], ...]
            'attention_mask': [[], ...]
                }
        where each sub list is of length self.sequence_length
        """

        def grouper(tokens: dict) -> dict:
            """Groups tokens into chunks of length `sequence_length`.

            Args:
                tokens (Dict[Tuple[str], List[int]]): A dictionary where each key is a token type
                and the value is a list of token values.

            Returns:
                Dict[str, List[List[int]]]: A dictionary where each key is a token type and the value
                 is a list of chunks of length `sequence_length`.
            """
            # Group all tokens by their keys
            all_tokens = {k: sum(map(lambda x: x[k], tokens), []) for k in tokens[0].keys()}
            # Calculate total number of tokens
            total_length = len(all_tokens[list(all_tokens.keys())[0]])
            # Adjust total length to be a multiple of sequence_length if necessary
            # if total_length >= sequence_length :
            #     total_length = (total_length // sequence_length ) * sequence_length
            # Group tokens into chunks
            result = {
                k: [t[i : i + self.sequence_length] for i in range(0, total_length, self.sequence_length)]
                for k, t in all_tokens.items()
            }
            return result

        dataset_base_path = os.path.dirname(self.inputlist_path)
        with open(self.inputlist_path) as f:
            raw_files = [os.path.join(dataset_base_path, x.strip()) for x in f.readlines()]

        data = []
        for file in raw_files:
            data.extend(open(file, encoding="utf-8").readlines())

        # read the paths to files that contain text
        if self.use_calibration and self.calibration_indices:
            max_index = max(self.calibration_indices)
            if max_index >= len(data):
                raise ValueError(
                    f"No calibration data available for index={max_index} provided "
                    f"in the inputlist. Available data length: {len(data)}"
                )
            # Use specified indices from the input list
            data = [data[idx] for idx in self.calibration_indices]

        tokens = list(map(self.tokenizer, data))
        # split by sequence_length
        grouped = grouper(tokens)
        # zero pad the last list/ group
        for k in grouped:
            pad_len = self.sequence_length - len(grouped[k][-1])
            grouped[k][-1] += [0] * pad_len
        # left pad the attention_mask by past_sequence_length
        for i in range(len(grouped["attention_mask"])):
            grouped["attention_mask"][i] = [0] * self.past_sequence_length + grouped["attention_mask"][i]
        return grouped
