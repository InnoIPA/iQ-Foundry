# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from os import PathLike
from typing import Any, Dict, List, Optional, Type, Union

from qti.aisw.tools.core.modules.api.backend.aic_backend import AicBackend
from qti.aisw.tools.core.modules.api.backend.backend import Backend
from qti.aisw.tools.core.modules.api.backend.cpu_backend import CpuBackend
from qti.aisw.tools.core.modules.api.backend.gpu_backend import GpuBackend
from qti.aisw.tools.core.modules.api.backend.htp_backend import HtpBackend
from qti.aisw.tools.core.modules.api.backend.htp_mcp_backend import HtpMcpBackend
from qti.aisw.tools.core.modules.api.backend.lpai_backend import LpaiBackend
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, QNNContextConfig, Target
from qti.aisw.tools.core.modules.api.utils.model_level_api import create_mlapi_target

_backend_mapping: Dict[BackendType, Type[Backend]] = {
    BackendType.CPU: CpuBackend,
    BackendType.GPU: GpuBackend,
    BackendType.HTP: HtpBackend,
    BackendType.HTP_MCP: HtpMcpBackend,
    BackendType.AIC: AicBackend,
    BackendType.LPAI: LpaiBackend,
}


def get_supported_backends() -> List[str]:
    """Returns a list of supported backend types.

    Returns:
        List[str]: A list of supported backend types as strings.
    """
    return list(_backend_mapping)


def get_backend_class(backend: BackendType) -> Type[Backend]:
    """Returns the backend class corresponding to the given backend type.

    Args:
        backend (BackendType): The type of the backend.

    Returns:
        Type[Backend]: The class of the backend corresponding to the given type.

    Raises:
        RuntimeError: If the backend type is not recognized.
    """
    if backend in _backend_mapping:
        return _backend_mapping[backend]
    raise RuntimeError(f"Backend {backend} not recognized")


def create_backend(
    backend: str | BackendType,
    target: Optional[Target],
    config_file: Optional[Union[str, PathLike]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    context_config: Optional[QNNContextConfig] = None,
) -> Backend:
    """Creates an instance of the specified backend type.

    Args:
        backend (BackendType): The type of the backend to create.
        target (Optional[Target]): The target for the backend.
        config_file (Optional[Union[str, PathLike]]): Path to the configuration file.
        config_dict (Optional[Dict[str, Any]]): Dictionary containing configuration parameters.
        context_config (Optional[QNNContextConfig]): Dictionary containing context configuration
                                                          parameters.
    Returns:
        Backend: An instance of the specified backend type.

    Raises:
        ValueError: If the backend type is unknown.
    """
    backend = BackendType(backend) if isinstance(backend, str) else backend
    mlapi_target = create_mlapi_target(target) if target else None

    backend_type = _backend_mapping.get(backend)
    if not backend_type:
        raise ValueError(f"Unknown backend type: {backend}")

    if backend_type is CpuBackend:
        # CPU does not support backend specific configs, so skip passing config file/dict
        return backend_type(target=mlapi_target, context_config=context_config)
    else:
        return backend_type(
            target=mlapi_target,
            config_file=config_file,
            config_dict=config_dict,
            context_config=context_config,
        )
