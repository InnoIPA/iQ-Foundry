# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.tools.core.utilities.data_processing import PostProcessor, TextRepresentation
from qti.aisw.tools.core.utilities.data_processing.utils import Helper
import numpy as np

class SquadPostProcessor(PostProcessor):
    """Predict answers for SQuAD dataset given start and end scores."""

    def __init__(self, do_unpacking: bool = False) -> None:
        """Initializes a new instance of the SQuAD postprocessor.

        Args:
            do_unpacking (bool): Whether to unpack data structures. Defaults to False.
        """
        self.do_unpacking = do_unpacking
        self.validate()

    def validate(self) -> None:
        """Validate the postprocessor parameters.

        Raises:
            ValueError: If do_unpacking is not a boolean.
            RuntimeError: If transformers package cannot be imported.
        """
        if not isinstance(self.do_unpacking, bool):
            raise ValueError("do_unpacking must be a boolean")
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        if transformers is None:
            raise RuntimeError(
                "transformers package is required for this postprocessor."
                f" Please install it to use {self.__class__.__name__}"
            )

    def validate_input(self, input_sample: TextRepresentation) -> TextRepresentation:
        """Validate the input sample.

        Args:
            input_sample (TextRepresentation): The input sample to be validated.

        Raises:
            ValueError: If metadata is missing or contains invalid keys.
        """
        required_metadata_keys = ["features"]

        # Only require packing_map if unpacking is enabled
        if self.do_unpacking:
            required_metadata_keys.append("packing_map")
        for meta_key in required_metadata_keys:
            if meta_key not in input_sample.metadata:
                raise ValueError(f"{meta_key} is a required in metadata to perform the processing")
        if len(input_sample.data) < 2:
            raise ValueError("input sample data must contain the start and end logits")
        return input_sample

    @PostProcessor.validate_input_output
    def execute(self, input_sample: TextRepresentation) -> TextRepresentation:
        """Execute the postprocessing on the given input sample.

        Args:
            input_sample (TextRepresentation): The input sample to be processed.

        Returns:
            TextRepresentation: The processed input sample.
        """
        start_logits, end_logits = input_sample.data
        input_idx = input_sample.idx
        features = input_sample.metadata["features"]
        transformers = Helper.safe_import_package("transformers", "4.44.0")  # noqa: F841
        from transformers.data.processors.squad import SquadResult

        # Normalize shapes to (384,) from (1, 384) etc.
        start_logits = np.array(start_logits).squeeze()
        end_logits = np.array(end_logits).squeeze()

        if self.do_unpacking:
            # info on which features were packed together
            # sls -> list of sequence lengths
            output_map = input_sample.metadata["packing_map"]
            feat_idx_and_sls = output_map[input_idx]
            feat_idxs = [x[0] for x in feat_idx_and_sls]
            sls = [x[1] for x in feat_idx_and_sls]

            # unpack
            packed_data = [start_logits.tolist(), end_logits.tolist()]
            unpacked_data = self.unpack(packed_data, sls)

            # add to "results" for each original feature that was packed
            results = []
            for data_idx, feat_idx in enumerate(feat_idxs):
                results.append(SquadResult(
                    features[feat_idx].unique_id,
                    list(unpacked_data[data_idx][0]),
                    list(unpacked_data[data_idx][1]),
                ))
        else:
            # Single output case
            results = [SquadResult(features[input_idx].unique_id, list(start_logits), list(end_logits))]
        input_sample.data = results
        return input_sample

    @staticmethod
    def unpack(data, sls):
        """Unpack the model output.

        Args:
            data (List[List[float]]): The packed model output.
                 Shape: [(384,), (384,)]
            sls (List[int]): The sequence lengths.

        Returns:
            List[List[List[float]]]: The unpacked model output.
        """
        # data is single model output [start_logits, end_logits]
        # sls is list of sequence lengths # [190, 192]
        # returns:
        #     list of [[start_logits, end_logits], ...]
        result = []
        offset = 0
        for sl in sls:
            res = [x[offset : offset + sl] for x in data]
            result.append(res)
            offset += sl
        return result
