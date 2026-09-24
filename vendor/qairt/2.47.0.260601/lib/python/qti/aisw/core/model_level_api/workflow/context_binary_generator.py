# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
from typing import List, Union

from qti.aisw.tools.core.modules.api.definitions import ModelConfig
from qti.aisw.core.model_level_api.executor.native_executor import NativeExecutor
from qti.aisw.core.model_level_api.executor.x86_subprocess_executor import (
    X86SubprocessExecutor,
)
from qti.aisw.core.model_level_api.workflow.workflow import Workflow, WorkflowMode
from qti.aisw.tools.core.modules.api.backend.backend import Backend

logger = logging.getLogger(__name__)


class ContextBinaryGenerator(Workflow):
    def __init__(
        self,
        backend: Backend,
        model: Union[ModelConfig, List[ModelConfig]],
        executor=None,
        sdk_path=None,
    ):
        super().__init__(backend, model, executor, sdk_path)
        workflow_mode = WorkflowMode.CONTEXT_BINARY_GENERATION
        self._backend.workflow_mode = workflow_mode

        if self._executor is None:
            target_default_executor_cls = (
                self._backend.target.get_default_executor_cls()
            )
            self._executor = target_default_executor_cls()

        if isinstance(self._model, list):
            if self._executor and not isinstance(
                self._executor, (NativeExecutor, X86SubprocessExecutor)
            ):
                raise RuntimeError(
                    "Context binary generation for multiple DLCs is only"
                    " supported on X86"
                )

        self._executor.setup(
            workflow_mode, self._backend, self._model, self._sdk_path, None, None
        )

    def generate(
        self,
        output_path="./output/",
        output_filename=None,
        backend_specific_filename=None,
        config=None,
    ):
        context_bin, backend_binary_path, profiling_data = (
            self._executor.generate_context_binary(
                config,
                self._backend,
                self._model,
                self._sdk_path,
                output_path,
                output_filename,
                backend_specific_filename,
            )
        )
        if profiling_data is not None:
            self._profiling_data.append(profiling_data)
        return context_bin, backend_binary_path
