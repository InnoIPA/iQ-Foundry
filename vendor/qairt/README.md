# Vendored QAIRT SDK

Qualcomm AI Runtime (QAIRT) `2.47.0.260601`, pruned to the linux-x86_64 host subset that
`--runtime qairt` needs. Shipped in-tree so `qc` mode converts fully offline, with no
Qualcomm AI Hub account and no device-side compilation.

## Layout

- `bin/` — `qairt-converter`, `qairt-quantizer`, `qairt-dlc-info`,
  `qnn-context-binary-generator`, `qnn-context-binary-utility`.
- `lib/` — the HTP backend libraries loaded during offline prepare (`libQnnHtp.so`,
  `libHtpPrepare.so`, `libQnnModelDlc.so`, `libQnnHtpNetRunExtensions.so`,
  `libQnnSystem.so`), plus five OS libraries the container image does not provide
  (`libpython3.10.so.1.0`, `libc++.so.1`, `libc++abi.so.1`, `libunwind.so.1`,
  `libatomic.so.1`). Without those five the SDK's native modules fail to load with a
  misleading "circular import" error.
- `lib/python/qti/aisw/` — converter/quantizer Python packages, with non-linux-x86_64
  platform directories removed.

## Target side

Nothing is pushed to the device. QLI2.0 already ships the QAIRT runtime
(`qnn-net-run`, `qnn-profile-viewer`, `libQnnHtp.so`, the v73 stub and DSP skel), so
`mAP` and `test` invoke the device's own binaries over adb.

## Environment

`tool/qairt_inference.py` sets `QNN_SDK_ROOT`, `PATH`, `LD_LIBRARY_PATH` and
`PYTHONPATH` per subprocess; no image or Dockerfile changes are required.
