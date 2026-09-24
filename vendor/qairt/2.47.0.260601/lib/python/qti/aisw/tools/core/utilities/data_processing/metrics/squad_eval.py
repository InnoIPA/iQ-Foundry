# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
# Source: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
################################################################################
import os
import tempfile

from qti.aisw.tools.core.utilities.data_processing import Representation
from qti.aisw.tools.core.utilities.data_processing.metrics import Metric
from qti.aisw.tools.core.utilities.data_processing.utils import Helper


class SquadEvaluation(Metric):
    """Class for metrics (F1 and exact) for SQuAD dataset."""

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
        max_answer_length: int = 30,
        n_best_size: int = 20,
        do_lower_case: bool = True,
        squad_version: int = 1,
        decimal_places: int = 6,
    ):
        """Initializes the SquadEvaluation instance.

        Args:
            tokenizer_model_name_or_path: str| os.PathLike: Can be
            - A string, the model id of a predefined tokenizer hosted inside a model repo on huggingface.co.
            - A string, the model id of a predefined tokenizer from huggingface.co (user-uploaded)
             and cache (e.g. "deepset/roberta-base-squad2")
            - A path to a directory containing vocabulary files required by the tokenizer,
                 for instance saved using the save_pretrained() method, e.g., ./my_model_directory/.

            max_answer_length (int): Maximum length of answers.
            n_best_size (int): Number of best answers to consider.
            do_lower_case (bool): Whether to convert text to lowercase.
            squad_version (int): The version of the SQuAD dataset (1 or 2).
            decimal_places (int): Precision for storing results as floats.
        """
        transformers = Helper.safe_import_package("transformers", "4.44.0")  # noqa: F841
        self.decimal_places = decimal_places
        self.tokenizer_model_name_or_path = tokenizer_model_name_or_path
        self.max_answer_length = max_answer_length
        self.n_best_size = n_best_size
        self.do_lower_case = do_lower_case
        self.squad_version = squad_version
        self.validate()
        self.all_results = []
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.tokenizer_model_name_or_path, do_lower_case=self.do_lower_case, use_fast=False
            )
        except Exception:
            raise RuntimeError(f"Failed to setup the tokenizer. {SquadEvaluation.TOKENIZER_INFO}")

    def validate(self) -> None:
        """Validates the input parameters.

        Raises:
            ValueError: If any of the following conditions are met:
            - The squad version is not 1 or 2.
            - `decimal_places` is not a positive integer.
            - `max_answer_length` is not a positive integer.
            - `n_best_size` is not a positive integer.
            - `do_lower_case` is not a boolean value.
            - `tokenizer_model_name_or_path` is not a valid string or path.
            - The tokenizer model path does not exist if it's a file path.

        """
        if self.squad_version not in [1, 2]:
            raise ValueError("Only squad versions 1 and 2 are supported")
        if not isinstance(self.decimal_places, int) or self.decimal_places < 1:
            raise ValueError("decimal_places must be a positive integer.")
        if not isinstance(self.max_answer_length, int) or self.max_answer_length < 1:
            raise ValueError("max_answer_length must be a positive integer.")
        if not isinstance(self.n_best_size, int) or self.n_best_size < 1:
            raise ValueError("n_best_size must be a positive integer.")
        if not isinstance(self.do_lower_case, bool):
            raise ValueError("do_lower_case must be a boolean.")
        if not isinstance(self.tokenizer_model_name_or_path, (str, os.PathLike)):
            raise ValueError("tokenizer model name or path must be a valid string or path.")

    def validate_input(self, input_sample: Representation) -> Representation:
        """Validate that the input sample contains required keys to calculate SQuAD metrics.

        Args:
            input_sample (Representation): The input sample to be validated

        Returns:
            Representation: The validated input sample

        Raises:
            RuntimeError: If any required key is missing in the input sample's metadata
        """
        required_keys = ["features", "examples"]
        missing_keys = [key for key in required_keys if key not in input_sample.metadata]
        if missing_keys:
            raise RuntimeError(
                f"Missing required keys in model_output.metadata: {missing_keys}."
                " These are required to calculate SQuAD metrics."
            )
        return input_sample

    @Metric.validate_input_output
    def calculate(self, model_output):
        """Calculate the SQuAD evaluation metrics for a given model output.

        Args:
            model_output (object): The output of the model to be evaluated.
                This object should contain the predictions made by the model on a test set.

        Returns:
            None (appends results to self.all_results)
        """
        # Setup the features and examples attributes for calculation of SQuAD metric
        if not hasattr(self, "features"):
            self.features = model_output.metadata["features"]
        if not hasattr(self, "examples"):
            self.examples = model_output.metadata["examples"]
        # flatten the postprocessed results into a single list of SQuAD results
        self.all_results.extend(model_output.data)

    def finalize(self) -> dict:
        """Finalizes the evaluation by computing the F1 and exact metrics.

        Returns:
            dict: A dictionary containing the F1 and exact metrics.
        """
        transformers = Helper.safe_import_package("transformers", "4.44.0")
        from transformers.data.metrics.squad_metrics import compute_predictions_logits

        results = {}

        if self.all_results:
            print(
                f"all results: {len(self.all_results)}, "
                f"examples: {len(self.examples)}, features: {len(self.features)}"
            )
            fd, temp_file_name = tempfile.mkstemp(suffix=".json", prefix="squad_metric_")
            os.close(fd)
            try:
                predictions = compute_predictions_logits(
                    self.examples,
                    self.features,
                    self.all_results,
                    self.n_best_size,
                    self.max_answer_length,
                    self.do_lower_case,
                    output_prediction_file=temp_file_name,
                    output_nbest_file=None,
                    output_null_log_odds_file=None,
                    verbose_logging=False,
                    version_2_with_negative=True if self.squad_version == 2 else False,
                    null_score_diff_threshold=0.0,
                    tokenizer=self.tokenizer,
                )

                full_results = transformers.data.metrics.squad_metrics.squad_evaluate(self.examples,
                                                                                     predictions)
                results["f1"] = round(full_results["f1"], self.decimal_places)
                results["exact"] = round(full_results["exact"], self.decimal_places)
                results["total"] = full_results["total"]
            except Exception as e:
                raise RuntimeError(f"Error during prediction computation: {e}")
            finally:
                # Clean up the temp file
                if os.path.exists(temp_file_name):
                    os.remove(temp_file_name)

        return results
