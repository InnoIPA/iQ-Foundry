# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import sys
import platform

if platform.system() == "Linux":
    if platform.machine() == "x86_64":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "linux-x86_64"))
    elif platform.machine() == "aarch64":
        if not (sys.version_info[0] == 3 and sys.version_info[1] == 12):
            raise NotImplementedError(
                f"On Linux aarch64, Python 3.12 is required. "
                f"Got: {sys.version_info[0]}.{sys.version_info[1]}"
            )
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "linux-aarch64-oe-gcc11.2")
        )
    else:
        raise NotImplementedError(
            "Unsupported OS Platform: {} {}".format(
                platform.system(), platform.machine()
            )
        )
elif platform.system() == "Windows":
    if "AMD64" in platform.processor() or "Intel64" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "windows-x86_64"))
    elif "ARMv8" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "windows-arm64ec"))
    else:
        cpu_isa = platform.processor().split()[0]
        raise NotImplementedError(
            "Unsupported OS Platform: {} {}".format(platform.system(), cpu_isa)
        )
else:
    raise NotImplementedError(
        "Unsupported OS Platform: {} {}".format(platform.system(), platform.machine())
    )

try:
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        import libPyIrGraph36 as ir_graph
        import libPyIrSerializer36 as qnn_ir
        import libPyIrJsonSerializer36 as json_serializer
        import libPyIrJsonDeserializer36 as json_deserializer
        import libPyIrQuantizer36 as ir_quantizer
        import libPyLightWeightIrQuantizer36 as light_weight_ir_quantizer
        import libPyIrEncodingsJsonSerializer36 as encodings_json_serializer
        import libPyBackendInfo36 as backend_info

        import libPyIrOptimizer36 as ir_optimizer
        import libPyBackendOpInfo36 as backend_opinfo

    elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
        import libPyIrGraph38 as ir_graph
        import libPyIrSerializer38 as qnn_ir
        import libPyIrJsonSerializer38 as json_serializer
        import libPyIrJsonDeserializer38 as json_deserializer
        import libPyIrQuantizer38 as ir_quantizer
        import libPyLightWeightIrQuantizer38 as light_weight_ir_quantizer
        import libPyIrEncodingsJsonSerializer38 as encodings_json_serializer
        import libPyBackendInfo38 as backend_info

        import libPyIrOptimizer38 as ir_optimizer
        import libPyBackendOpInfo38 as backend_opinfo

    elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
        import libPyIrGraph310 as ir_graph
        import libPyIrSerializer310 as qnn_ir
        import libPyIrJsonSerializer310 as json_serializer
        import libPyIrJsonDeserializer310 as json_deserializer
        import libPyIrQuantizer310 as ir_quantizer
        import libPyLightWeightIrQuantizer310 as light_weight_ir_quantizer
        import libPyIrEncodingsJsonSerializer310 as encodings_json_serializer
        import libPyBackendInfo310 as backend_info

        import libPyIrOptimizer310 as ir_optimizer
        import libPyBackendOpInfo310 as backend_opinfo

    else:
        import libPyIrGraph as ir_graph
        import libPyIrSerializer as qnn_ir
        import libPyIrJsonSerializer as json_serializer
        import libPyIrJsonDeserializer as json_deserializer
        import libPyIrQuantizer as ir_quantizer
        import libPyLightWeightIrQuantizer as light_weight_ir_quantizer
        import libPyIrEncodingsJsonSerializer as encodings_json_serializer
        import libPyBackendInfo as backend_info

        import libPyIrOptimizer as ir_optimizer
        import libPyBackendOpInfo as backend_opinfo

except ImportError as e:
    try:
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            from . import libPyIrGraph36 as ir_graph
            from . import libPyIrSerializer36 as qnn_ir
            from . import libPyIrJsonSerializer36 as json_serializer
            from . import libPyIrJsonDeserializer36 as json_deserializer
            from . import libPyIrQuantizer36 as ir_quantizer
            from . import libPyLightWeightIrQuantizer36 as light_weight_ir_quantizer
            from . import libPyIrEncodingsJsonSerializer36 as encodings_json_serializer
            from . import libPyBackendInfo36 as backend_info

            from . import libPyIrOptimizer36 as ir_optimizer
            from . import libPyBackendOpInfo36 as backend_opinfo

        elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
            from . import libPyIrGraph38 as ir_graph
            from . import libPyIrSerializer38 as qnn_ir
            from . import libPyIrJsonSerializer38 as json_serializer
            from . import libPyIrJsonDeserializer38 as json_deserializer
            from . import libPyIrQuantizer38 as ir_quantizer
            from . import libPyLightWeightIrQuantizer38 as light_weight_ir_quantizer
            from . import libPyIrEncodingsJsonSerializer38 as encodings_json_serializer
            from . import libPyBackendInfo38 as backend_info

            from . import libPyIrOptimizer38 as ir_optimizer
            from . import libPyBackendOpInfo38 as backend_opinfo

        elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
            from . import libPyIrGraph310 as ir_graph
            from . import libPyIrSerializer310 as qnn_ir
            from . import libPyIrJsonSerializer310 as json_serializer
            from . import libPyIrJsonDeserializer310 as json_deserializer
            from . import libPyIrQuantizer310 as ir_quantizer
            from . import libPyLightWeightIrQuantizer310 as light_weight_ir_quantizer
            from . import libPyIrEncodingsJsonSerializer310 as encodings_json_serializer
            from . import libPyBackendInfo310 as backend_info

            from . import libPyIrOptimizer310 as ir_optimizer
            from . import libPyBackendOpInfo310 as backend_opinfo

        else:
            from . import libPyIrGraph as ir_graph
            from . import libPyIrSerializer as qnn_ir
            from . import libPyIrJsonSerializer as json_serializer
            from . import libPyIrJsonDeserializer as json_deserializer
            from . import libPyIrQuantizer as ir_quantizer
            from . import libPyLightWeightIrQuantizer as light_weight_ir_quantizer
            from . import libPyIrEncodingsJsonSerializer as encodings_json_serializer
            from . import libPyBackendInfo as backend_info

            from . import libPyIrOptimizer as ir_optimizer
            from . import libPyBackendOpInfo as backend_opinfo

    except ImportError:
        raise e

# DlModelTools will only be present in common/ for QNN builds
try:
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        import libDlModelToolsPy36 as modeltools
    elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
        import libDlModelToolsPy38 as modeltools
    elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
        import libDlModelToolsPy310 as modeltools
    else:
        import libDlModelToolsPy as modeltools
except ImportError as e:
    try:
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            from . import libDlModelToolsPy36 as modeltools
        elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
            from . import libDlModelToolsPy38 as modeltools
        elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
            from . import libDlModelToolsPy310 as modeltools
        else:
            from . import libDlModelToolsPy as modeltools
    except ImportError:
        pass
