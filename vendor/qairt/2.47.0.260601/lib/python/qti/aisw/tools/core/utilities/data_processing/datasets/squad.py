# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
from collections import defaultdict, namedtuple
from queue import Queue
from typing import Literal, Optional

import numpy as np
from qti.aisw.tools.core.utilities.data_processing import NDArrayRepresentation
from qti.aisw.tools.core.utilities.data_processing.datasets import IndexableDataset
from qti.aisw.tools.core.utilities.data_processing.datasets._spfhp import pack_using_spfhp
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


PackedInputs = namedtuple(
    "PackedInputs", ["input_ids", "attention_mask", "token_type_ids", "input_position_ids"]
)


class SQUADDataset(IndexableDataset):
    """This class is responsible for loading and preprocessing the SQUAD dataset.

    Attributes:
        tokenizer_model_name_or_path (str | os.PathLike): The name or path to the model used for tokenization.
        annotation_path (Optional[os.PathLike | str]): Path to the SQUAD annotation file. Defaults to None.
        calibration_path (Optional[os.PathLike | str]): Path to the SQUAD calibration file. Defaults to None.
        max_samples (Optional[int]): The maximum number of samples to load. Defaults to None.
        use_calibration (Optional[bool]): Whether to use calibration data or not. Defaults to False.
        max_seq_length (int): The maximum sequence length. Defaults to 384.
        max_query_length (int): The maximum query length. Defaults to 64.
        doc_stride (int): The document stride. Defaults to 128.
        threads (int): The number of threads to use. Defaults to 8.
        do_lower_case (bool): Whether to perform lower-casing on the data. Defaults to True.
        model_inputs_count (int): The number of input fields in the PackedInputs tuple. Defaults to 2.
        use_packing_strategy (bool): Whether to pack features or not. Defaults to False.
        max_sequence_per_pack (int): The maximum number of sequences per pack. Defaults to 3.
        mask_type (Optional[Literal['boolean', 'compressed']]): The type of mask to use. Defaults to None.
        compressed_mask_length (Optional[int]): The length of the compressed mask. Defaults to None.
        squad_version (int): The version of the SQUAD dataset. Defaults to 1.
    """

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
        annotation_path: Optional[os.PathLike | str] = None,
        calibration_path: Optional[os.PathLike | str] = None,
        max_samples: Optional[int] = None,
        use_calibration: Optional[bool] = False,
        max_seq_length: int = 384,
        max_query_length: int = 64,
        doc_stride: int = 128,
        threads: int = 8,
        do_lower_case: bool = True,
        model_inputs_count: int = 2,
        use_packing_strategy: bool = False,
        max_sequence_per_pack: int = 3,
        mask_type: Optional[Literal["boolean", "compressed"]] = None,
        compressed_mask_length: Optional[int] = None,
        squad_version: Literal[1, 2] = 1,
    ):
        """Initialize the SQUAD dataset.

        Args:
            tokenizer_model_name_or_path (str): The name or path to the model used for tokenization.
            annotation_path (Optional[os.PathLike | str]): Path to the SQUAD annotation file.
             Defaults to None.
            calibration_path (Optional[os.PathLike | str]): Path to the SQUAD calibration file.
             Defaults to None.
            max_samples (Optional[int]): The maximum number of samples to load.
             Defaults to None.
            use_calibration (Optional[bool]): Whether to use calibration data or not.
             Defaults to False.
            max_seq_length (int): The maximum sequence length. Defaults to 384.
            max_query_length (int): The maximum query length. Defaults to 64.
            doc_stride (int): The document stride. Defaults to 128.
            threads (int): The number of threads to use. Defaults to 8.
            do_lower_case (bool): Whether to perform lower-casing on the data.
             Defaults to True.
            model_inputs_count (int): The number of input fields in the PackedInputs tuple.
             Defaults to 2.
            use_packing_strategy (bool): Whether to pack features or not. Defaults to False.
            max_sequence_per_pack (int): The maximum number of sequences per pack.
             Defaults to 3.
            mask_type (Optional[Literal['boolean', 'compressed']]): The type of mask to use.
             Defaults to None.
            compressed_mask_length (Optional[int]): The length of the compressed mask.
             Defaults to None.
            squad_version (int): The version of the SQUAD dataset. Defaults to 1.
        """
        self.tokenizer_model_name_or_path = tokenizer_model_name_or_path
        self.annotation_path = annotation_path
        self.calibration_path = calibration_path
        self.max_samples = max_samples
        self.use_calibration = use_calibration
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.threads = threads
        self.do_lower_case = do_lower_case
        self.model_inputs_count = model_inputs_count
        self.use_packing_strategy = use_packing_strategy
        self.max_sequence_per_pack = max_sequence_per_pack
        self.mask_type = mask_type
        self.compressed_mask_length = compressed_mask_length
        self.squad_version = squad_version
        self.validate()
        self._setup()

    def validate(self):
        """Validate the provided parameters."""
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        if transformers is None:
            raise ImportError("transformers library is required to use SQUAD dataset")

        if self.annotation_path and not os.path.exists(self.annotation_path):
            raise FileNotFoundError(
                f"Annotation file provided does not exist. provided: {self.annotation_path}"
            )
        if self.calibration_path and not os.path.exists(self.calibration_path):
            raise FileNotFoundError(
                f"Calibration file provided does not exist. provided: {self.calibration_path}"
            )
        if not isinstance(self.use_calibration, bool):
            raise ValueError("use_calibration must be a boolean")

        if self.use_calibration and self.calibration_path is None:
            raise ValueError("When use_calibration is set to True, user must provide calibration_path")

        if type(self.squad_version) is not int or self.squad_version not in [1, 2]:
            raise ValueError("squad version must be one of the following: 1 or 2")

        if self.mask_type is not None and self.mask_type not in ["compressed", "boolean"]:
            raise ValueError("mask type must be one of the following: compressed or boolean")

        if not isinstance(self.use_packing_strategy, bool):
            raise ValueError("use_packing_strategy must be a boolean")

        if not isinstance(self.do_lower_case, bool):
            raise ValueError("do_lower_case must be a boolean")

        if self.max_samples is not None:
            if type(self.max_samples) is not int:
                raise ValueError("max_samples must be a integer")
            if self.max_samples == 0:
                raise ValueError("max_samples cannot be 0. Provide valid values")

        if type(self.max_sequence_per_pack) is not int or self.max_sequence_per_pack < 1:
            raise ValueError("max_sequence_per_pack must be a non negative intger")

        if type(self.model_inputs_count) is not int or self.model_inputs_count < 1:
            raise ValueError("model_inputs_count must be a non negative intger")

        if type(self.doc_stride) is not int or self.doc_stride < 1:
            raise ValueError("doc_stride must be a non negative intger")

        if type(self.max_query_length) is not int or self.max_query_length < 1:
            raise ValueError("max_query_length must be a non negative intger")

        if type(self.max_seq_length) is not int or self.max_seq_length < 1:
            raise ValueError("max_seq_length must be a non negative intger")

        if self.compressed_mask_length and (
            type(self.compressed_mask_length) is not int or self.compressed_mask_length < 1
        ):
            raise ValueError("compressed_mask_length must be a non negative intger")

    def _setup(self):
        """Set up the Squad dataset.

        This method initializes the necessary components for processing the SQuAD dataset,
        including the data file path, processor, and tokenizer.
        """
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        if self.use_calibration:
            self.data_file_path = self.calibration_path
        else:
            self.data_file_path = self.annotation_path
        if self.squad_version == 1:
            self.processor = transformers.data.processors.squad.SquadV1Processor()
        elif self.squad_version == 2:
            self.processor = transformers.data.processors.squad.SquadV2Processor()
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.tokenizer_model_name_or_path, do_lower_case=self.do_lower_case, use_fast=False
            )
        except Exception:
            raise RuntimeError(f"Failed to setup the tokenizer. {self.TOKENIZER_INFO}")

        self.dataset = self.prepare_data()

    def prepare_data(self) -> list[NDArrayRepresentation]:
        """Prepares the data for training or evaluation.

        This method loads the examples from the dataset file, converts them to features,
        and packs them according to the specified packing strategy. The packed features are
        then returned as a list of `NDArrayRepresentation` objects.

        Returns:
            list[NDArrayRepresentation]: A list of preprocessed data points.
        """
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        examples = self.processor.get_dev_examples("", filename=self.data_file_path)
        if self.max_samples not in [None, -1]:
            examples = examples[: self.max_samples]
        input_names = list(PackedInputs._fields)
        if not self.use_packing_strategy:
            input_names.remove("input_position_ids")
        model_inputs = input_names[: self.model_inputs_count]

        features = transformers.squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            threads=self.threads,
        )

        self.metadata = {"features": features, "examples": examples, "tokenizer": self.tokenizer}
        if self.use_packing_strategy:
            # eval_features is of type PackedInputs after packing
            features, packing_map = self.do_packing_strategy(
                features=features,
                max_sequence_length=self.max_seq_length,
                max_sequence_per_pack=self.max_sequence_per_pack,
                mask_type=self.mask_type,
                compressed_mask_length=self.compressed_mask_length,
            )
            if not self.use_calibration:
                # do not save the mapping file for calibration data
                self.metadata["packing_map"] = packing_map

        processed_data = []
        for index, feature in enumerate(features):
            preprocessed_input = []
            for model_input in model_inputs:
                if (
                    model_input == "attention_mask"
                    and self.mask_type
                    and self.mask_type.lower().startswith("bool")
                ):
                    preprocessed_input.append(np.array(getattr(feature, model_input), dtype="bool"))
                else:
                    preprocessed_input.append(np.array(getattr(feature, model_input)))
            processed_data.append(
                NDArrayRepresentation(data=preprocessed_input, metadata=self.metadata, idx=index)
            )
        return processed_data

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return sum(1 for i in self.dataset)

    def __getitem__(self, index) -> NDArrayRepresentation:
        """Get an item from the dataset at a given index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Any: The item at the specified index.
        """
        return self.dataset[index]

    @staticmethod
    def do_packing_strategy(
        features: list,
        max_sequence_per_pack: int,
        max_sequence_length: int,
        mask_type: str,
        compressed_mask_length: Optional[int] = None,
    ) -> tuple[list[PackedInputs], dict[str, list[int]]]:
        """Applies packing strategy to the given features based on their sequence lengths.

        Args:
            features (List[SQUADFeature]): The input features.
            max_sequence_per_pack (int): The maximum number of sequences to pack together.
            max_sequence_length (int): The maximum length of a sequence.
            mask_type (str): Type of attention mask.
            compressed_mask_length (Optional[int], optional): Length of the compressed mask. Defaults to None.

        Returns:
            Tuple[List[PackedSQUADFeature], Dict[str, List[int]]]: The packed features and their packing map.
        """
        logging.debug("Getting Strategy set and frequency")
        # generate histogram
        sequence_lengths = [sum(f.attention_mask) for f in features]
        histogram = [0] * max_sequence_length
        for sl in sequence_lengths:
            histogram[sl - 1] += 1

        # get packing strategy

        strategy_set, strategy_repeat_count = pack_using_spfhp(
            np.array(histogram), max_sequence_length, max_sequence_per_pack
        )

        # pack and write packed sequences
        logging.debug("Packing the Features")
        packed_inputs, packing_map = SQUADDataset.do_packing(
            features,
            sequence_lengths,
            strategy_set=strategy_set,
            strategy_repeat_count=strategy_repeat_count,
            max_sequence_length=max_sequence_length,
            mask_type=mask_type,
            compressed_mask_length=compressed_mask_length,
        )

        logging.debug(f"Number of sequences before packing: {len(features)}")
        logging.debug(f"Number of sequences after packing : {len(packed_inputs)}")
        return packed_inputs, packing_map

    @staticmethod
    def do_packing(
        features: list[tuple],
        sequence_lengths: list[int],
        strategy_set: list[list[int]],
        strategy_repeat_count: list[int],
        max_sequence_length: int,
        mask_type: str,
        compressed_mask_length: int = None,
    ) -> tuple[list[PackedInputs], dict[int, list[tuple[int, int]]]]:
        """Packs features based on sequence lengths and strategies.

        Args:
            features (List[Tuple]): Features to be packed.
            sequence_lengths (List[int]): Corresponding sequence lengths for each feature.
            strategy_set (List[List[int]]): Strategy set defining which sequence lengths are packed together.
            strategy_repeat_count (List[int]): Repeat count for each strategy.
            max_sequence_length (int): Maximum sequence length in the dataset.
            mask_type (str): Mask type used during packing.
            compressed_mask_length (int, optional): Compressed mask length. Defaults to None.

        Returns:
            Tuple[List[PackedInputs], Dict[int, List[Tuple[int, int]]]]: Packed inputs and output map.
        """
        # create queues for each Sequence Length
        # and fill them up with the respective features (along with their index)
        features_by_sl = [Queue() for _ in range(max_sequence_length + 1)]
        for i, (feat, sl) in enumerate(zip(features, sequence_lengths)):
            features_by_sl[sl].put((i, feat))

        # store which features (and what sl) were packed into which directory
        # key: output dir (idx)
        # value: list of input feat (idx), and sl
        # example, {0:[(5001, 192), (5600, 191)]}
        output_map = defaultdict(list)
        output_idx = 0

        # iterate over strategy
        # and do the packing
        all_packed_inputs = []
        for group, group_freq in zip(strategy_set, strategy_repeat_count):
            # group (list) contains the SL that has to be packed together
            # pack it "group_freq" times
            for _ in range(group_freq):
                # get data to be packed together
                to_pack = []
                for sl in group:  # group: (192,190)
                    idx, feat = features_by_sl[sl].get_nowait()
                    to_pack.append(feat)
                    # store which directories are packed together
                    output_map[output_idx].append((idx, sl))

                packed_inputs = SQUADDataset.pack_features(
                    to_pack,
                    group,
                    max_sequence_length,
                    mask_type=mask_type,
                    compressed_mask_length=compressed_mask_length,
                )
                all_packed_inputs.append(packed_inputs)
                output_idx += 1

        # Verify if packed all features
        for i in range(max_sequence_length + 1):
            assert features_by_sl[i].empty()

        return all_packed_inputs, output_map

    @staticmethod
    def pack_features(
        feat_list: list["PackedFeatures"],  # noqa: F821
        sls: list[int],
        max_sl: int = 384,
        mask_type: Optional[str] = None,
        compressed_mask_length: Optional[int] = None,
    ) -> PackedInputs:
        """Pack together the provided features.

        This function takes a list of `PackedFeatures` instances, their corresponding sequence lengths
        (`sls`), and other parameters. It packs the input IDs, token type IDs, position IDs, and attention
        mask into a single instance of `PackedInputs`.

        Args:
            feat_list (List[PackedFeatures]): A list of `PackedFeatures` instances to be packed together.
            sls (List[int]): The corresponding sequence lengths for each feature in `feat_list`.
            max_sl (int, optional): The maximum sequence length. Defaults to 384.
            mask_type (str, optional): The type of attention mask to use. Can be either 'compressed'
             or None. Defaults to None.
            compressed_mask_length (int, optional): The length of the compressed attention mask.
             Defaults to None.

        Returns:
            PackedInputs: A `PackedInputs` instance containing the packed features.
        """
        input_ids = np.concatenate([feat.input_ids[:sl] for feat, sl in zip(feat_list, sls)])
        token_type_ids = np.concatenate([feat.token_type_ids[:sl] for feat, sl in zip(feat_list, sls)])

        # Convert to int64 (assuming input IDs and token type IDs are numpy arrays)
        input_ids = input_ids.astype(np.int64)
        token_type_ids = token_type_ids.astype(np.int64)

        # create input_position_ids
        position_ids = np.concatenate([np.arange(sl, dtype=np.int64) for sl in sls])

        # Padding
        pad_len = max_sl - sum(sls)
        assert pad_len >= 0
        assert len(input_ids) == len(token_type_ids) == len(position_ids)

        if mask_type and mask_type.lower() == "compressed":  # Compressed mask
            attention_mask = np.array(sls)
            if compressed_mask_length is not None:
                assert len(attention_mask) <= compressed_mask_length
                attention_mask = np.pad(attention_mask, [0, compressed_mask_length - len(attention_mask)])
        else:  # 2D mask
            attention_mask = SQUADDataset.gen_2d_mask(sls, pad_len)

        input_ids = np.pad(input_ids, [0, pad_len])
        token_type_ids = np.pad(token_type_ids, [0, pad_len])
        position_ids = np.pad(position_ids, [0, pad_len])

        return PackedInputs(input_ids, attention_mask, token_type_ids, position_ids)

    @staticmethod
    def gen_2d_mask(sequence_lengths: list[int], padding_length: int) -> np.ndarray:
        """Generate a 2D mask for sequence inputs.

        The mask is created as a block diagonal matrix where each row corresponds to a sequence.
        Non-zero elements indicate valid attention values, while zero elements indicate padded
         or invalid attention values.

        Args:
            sequence_lengths (list[int]): List of sequence lengths for each input sequence.
            padding_length (int): Additional padding length to be added to the mask.

        Returns:
            np.ndarray: A 2D numpy array representing the attention mask.

        Example:
            >>> sequence_lengths = [1, 2]
            >>> padding_length = 0
            >>> gen_2d_mask(sequence_lengths, padding_length)
            array([[1, 0, 0],
                   [0, 1, 1],
                   [0, 1, 1]])
        """
        mask = np.concatenate([[i] * sl for i, sl in enumerate(sequence_lengths)])[np.newaxis, :]
        attention_mask = 1 * np.equal(mask, mask.transpose())
        attention_mask = attention_mask.astype(np.int64)
        attention_mask = np.pad(attention_mask, [0, padding_length])
        return attention_mask
