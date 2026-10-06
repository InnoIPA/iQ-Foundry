# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import platform
import sys

def _is_oelinux_platform():
    import platform

    if platform.system() != "Linux" or platform.machine() != "aarch64":
        return False

    try:
        with open("/etc/os-release", "r") as f:
            os_release = f.read().lower()
            oe_identifiers = ["qcom", "yocto", "poky", "openembedded", "oe-linux", "ubuntu"]

            for line in os_release.splitlines():
                if any(field in line for field in ["id=", "name=", "pretty_name="]):
                    if any(identifier in line for identifier in oe_identifiers):
                        return True
            return False
    except FileNotFoundError:
        return False


if platform.system() == "Linux":
    if platform.machine() == "x86_64":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'linux-x86_64'))
    elif _is_oelinux_platform():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'linux-aarch64-oe-gcc11.2'))
    else:
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))
elif platform.system() == "Windows":
    if "AMD64" in platform.processor() or "Intel64" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-x86_64'))
    elif "ARMv8" in platform.processor():
        os.add_dll_directory(os.path.join(os.environ["QNN_SDK_ROOT"], "lib", "arm64x-windows-msvc"))
        os.environ['PATH'] = os.path.join(os.environ["QNN_SDK_ROOT"], "lib", "arm64x-windows-msvc") + os.pathsep + os.environ['PATH']
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-arm64ec'))
    else:
        cpu_isa = platform.processor().split()[0]
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), cpu_isa))
else:
    raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), platform.machine()))

try:
    if sys.version_info[0] == 3 and sys.version_info[1] == 6:
        import libPyNetRun36 as py_net_run
    elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
        import libPyNetRun38 as py_net_run
    elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
        import libPyNetRun310 as py_net_run
    elif sys.version_info[0] == 3 and sys.version_info[1] == 12:
        import libPyNetRun as py_net_run
    else:
        py_net_run=None
        raise ImportError(f"Unsupported Python version: {sys.version_info[0]}.{sys.version_info[1]}")
except ImportError as e:
    try:
        if sys.version_info[0] == 3 and sys.version_info[1] == 6:
            from . import libPyNetRun36 as py_net_run
        elif sys.version_info[0] == 3 and sys.version_info[1] == 8:
            from . import libPyNetRun38 as py_net_run
        elif sys.version_info[0] == 3 and sys.version_info[1] == 10:
            from . import libPyNetRun310 as py_net_run
        elif sys.version_info[0] == 3 and sys.version_info[1] == 12:
            from . import libPyNetRun as py_net_run
        else:
            py_net_run=None
            raise ImportError(f"Unsupported Python version: {sys.version_info[0]}.{sys.version_info[1]}")
    except ImportError:
        py_net_run=None
        print("Cannot import pybind library: libPyNetRun")
        raise e
