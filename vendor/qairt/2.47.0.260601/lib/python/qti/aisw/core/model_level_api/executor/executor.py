#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from abc import ABC, abstractmethod


class Executor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, workflow_mode, backend, model, sdk_root, config, output_dir):
        pass

    @abstractmethod
    def run_inference(
        self, config, backend, model, sdk_root, input_data, output_dir, graph_name
    ):
        pass

    @abstractmethod
    def generate_context_binary(self, config, backend, model, sdk_root, output_path,
                                output_filename, backend_specific_filename):
        pass

    def teardown(self, backend, sdk_root, config, output_dir):
        pass
