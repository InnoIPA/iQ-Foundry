# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from qti.aisw.converters.common import backend_info
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils.converter_utils import log_warning
import re
import argparse
import sys

class BackendInfo(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(BackendInfo.ArgParser, self).__init__(**kwargs)
            backend_group = self.add_argument_group(title='Backend Options')
            backend_group.add_argument('--backend', type=str, default="HTP",
                                       help=argparse.SUPPRESS)
            backend_group.add_argument('--target_backend', dest='backend', type=str, default="HTP",
                                       help='Use this option to specify the backend on which the model needs to run.\n'
                                            'Providing this option will generate a graph optimized for the given backend '
                                            'and this graph may not run on other backends. The default backend is HTP.\n'
                                            'Supported backends are ' + ",".join(backend_info.supported_backends()) + '.')
            backend_group.add_argument('--soc_model', type=str, default="",
                                       help=argparse.SUPPRESS)
            backend_group.add_argument('--target_soc_model', dest='soc_model', type=str, default="",
                                       help='Use this option to specify the SOC on which the model needs to run.\n'
                                            'This can be found from SOC info of the device and it starts with strings '
                                            'such as SDM, SM, QCS, IPQ, SA, QC, SC, SXR, SSG, STP, QRB, or AIC.\n'
                                            'NOTE: --target_backend option must be provided to use --target_soc_model option.')

    @staticmethod
    def get_instance(backend : str, soc_model : str = ""):
        backend_info_obj = None

        if '--backend' in sys.argv:
            log_warning("--backend option is deprecated, use --target_backend.")
        if '--soc_model' in sys.argv:
            log_warning("--soc_model option is deprecated, use --target_soc_model.")

        if soc_model and not backend:
            raise Exception("soc_model is provided without specifying backend")

        # If backend is not provided, Backend Info object should not be created
        if not backend:
            return backend_info_obj

        # Check whether backend is supported
        if not backend_info.is_backend_supported(backend):
            raise Exception("Backend {} is not supported. Supported backends are {}"
                            .format(backend, ",".join(backend_info.supported_backends())))

        # Check whether soc_model is supported
        if soc_model and not backend_info.is_soc_model_supported(soc_model):
                raise Exception("SOC model {} is not supported.".format(soc_model))

        # If backend is not received, Backend Info object should not be created
        if not backend:
            return backend_info_obj

        # Create BackendInfo object
        backend_info_obj = backend_info.PyBackendInfo(backend, soc_model)

        # soc_model can be optional depending on backend requirement. Hence, validate it
        # using the created object.
        if not backend_info_obj.is_backend_supported_in_soc():
            if not soc_model:
                raise Exception("Backend {} requires SOC model but not received."
                                .format(backend))
            raise Exception("Backend {} is not supported by the SOC Model {}."
                            .format(backend, soc_model))
        return backend_info_obj

    @staticmethod
    def validate_dlc_metadata(metadata, backend=None, soc_model=None):
        # validate backend
        backend_option = re.findall('backend=\w+', metadata)
        if backend_option and backend:
            backend_metadata = backend_option[0].split('=')[1]
            if backend_metadata != backend:
                log_warning("{} is provided to backend option but input DLC was generated"
                            " using backend {}".format(backend, backend_metadata))

        # validate soc_model
        soc_model_option = re.findall('soc_model=\w+', metadata)
        if soc_model_option and soc_model:
            soc_model_metadata = soc_model_option[0].split('=')[1]
            if soc_model_metadata != soc_model:
                log_warning("{} is provided to soc_model option but input DLC was generated"
                " using soc_model {}".format(soc_model, soc_model_metadata))
