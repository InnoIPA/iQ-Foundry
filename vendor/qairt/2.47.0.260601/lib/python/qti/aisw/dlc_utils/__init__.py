#=============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

import os
import sys
import platform

if platform.system() == "Linux":
    if platform.machine() == "x86_64":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'linux-x86_64'))
    elif platform.machine() == "aarch64":
        if not (sys.version_info[0] == 3 and sys.version_info[1] == 12):
            raise NotImplementedError(
                f"On Linux aarch64, Python 3.12 is required. "
                f"Got: {sys.version_info[0]}.{sys.version_info[1]}"
            )
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'linux-aarch64-oe-gcc11.2'))
    else:
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))
elif platform.system() == "Windows":
    if "AMD64" in platform.processor() or "Intel64" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-x86_64'))
    elif "ARMv8" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-arm64ec'))
    else:
        cpu_isa = platform.processor().split()[0]
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), cpu_isa))
else:
    raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))

try:
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        from . import libDlModelToolsPy36 as modeltools
        from . import libDlContainerPy36 as dlcontainer
    elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
        from . import libDlModelToolsPy38 as modeltools
        from . import libDlContainerPy38 as dlcontainer
    elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
        from . import libDlModelToolsPy310 as modeltools
        from . import libDlContainerPy310 as dlcontainer
    else:
        from . import libDlModelToolsPy as modeltools
        from . import libDlContainerPy as dlcontainer

except ImportError as ie1:
    try:
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            import libDlModelToolsPy36 as modeltools
            import libDlContainerPy36 as dlcontainer
        elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
            import libDlModelToolsPy38 as modeltools
            import libDlContainerPy38 as dlcontainer
        elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
            import libDlModelToolsPy310 as modeltools
            import libDlContainerPy310 as dlcontainer
        else:
            import libDlModelToolsPy as modeltools
            import libDlContainerPy as dlcontainer
    except ImportError:
        raise ie1

from qti.aisw.converters.common import ir_graph