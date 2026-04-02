#!/usr/bin/env bash
# Copyright (c) 2025 Innodisk Corp.
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[error] source this script instead: source setup.sh" >&2
    exit 1
fi

IQ_CV_QAI_HUB_STATUS="Not checked"
IQ_CV_ADB_STATUS="Not checked"

_iq_cv_setup_info() {
    printf '[setup] %s\n' "$*"
}

_iq_cv_setup_error() {
    printf '[error] %s\n' "$*" >&2
}

_iq_cv_setup_require_command() {
    local cmd_name="$1"
    local error_message="$2"
    if ! command -v "${cmd_name}" >/dev/null 2>&1; then
        _iq_cv_setup_error "${error_message}"
        return 1
    fi
}

_iq_cv_setup_require_sudo() {
    local reason="$1"
    _iq_cv_setup_info "sudo password prompt incoming: ${reason}"
    sudo -v
}

_iq_cv_setup_install_uv() {
    if command -v uv >/dev/null 2>&1; then
        return 0
    fi
    _iq_cv_setup_require_command \
        curl \
        "curl is required to install uv. Install curl and re-source setup.sh." || return 1
    _iq_cv_setup_info "Installing uv because it is required for the tutorial virtual environment."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        _iq_cv_setup_error "uv installation failed."
        return 1
    fi
    export PATH="${HOME}/.local/bin:${PATH}"
    _iq_cv_setup_require_command \
        uv \
        "uv was installed but is still not on PATH. Add ~/.local/bin to PATH and re-source setup.sh." || return 1
}

_iq_cv_setup_sync_host_deps() {
    local requirements_file="${IQ_CV_SCRIPT_DIR}/requirements/host.txt"
    local stamp_file="${IQ_CV_SCRIPT_DIR}/.venv/.iq_cv_host_requirements.sha256"
    local requirements_hash

    requirements_hash="$(sha256sum "${requirements_file}" | awk '{print $1}')"
    if [[ -f "${stamp_file}" ]] && [[ "$(<"${stamp_file}")" == "${requirements_hash}" ]]; then
        _iq_cv_setup_info "Host Python dependencies already match requirements/host.txt."
        return 0
    fi

    _iq_cv_setup_info "Installing host Python dependencies from requirements/host.txt."
    if ! uv pip install -r "${requirements_file}"; then
        _iq_cv_setup_error "Failed to install host Python dependencies."
        return 1
    fi
    printf '%s\n' "${requirements_hash}" > "${stamp_file}"
}

_iq_cv_setup_install_host_packages() {
    local packages=()
    if ! command -v adb >/dev/null 2>&1; then
        packages+=(adb)
    fi
    if [[ ! -f /lib/udev/rules.d/51-android.rules ]] && [[ ! -f /usr/lib/udev/rules.d/51-android.rules ]]; then
        packages+=(android-sdk-platform-tools-common)
    fi
    if (( ${#packages[@]} == 0 )); then
        return 0
    fi
    _iq_cv_setup_require_sudo \
        "installing required ADB host packages for Step 2: ${packages[*]}" || return 1
    if ! sudo apt update; then
        _iq_cv_setup_error "apt update failed."
        return 1
    fi
    if ! sudo apt install -y "${packages[@]}"; then
        _iq_cv_setup_error "Failed to install required host packages: ${packages[*]}."
        return 1
    fi
}

_iq_cv_setup_adb_device_count() {
    adb devices | awk 'NR > 1 && $2 == "device" {count++} END {print count + 0}'
}

_iq_cv_setup_verify_adb_target() {
    local adb_device_count=0

    if ! command -v adb >/dev/null 2>&1; then
        IQ_CV_ADB_STATUS="Not available"
        _iq_cv_setup_info "ADB is not installed or not on PATH."
        return 0
    fi

    adb start-server >/dev/null 2>&1 || true
    adb_device_count="$(_iq_cv_setup_adb_device_count)"
    if [[ "${adb_device_count}" -eq 0 ]]; then
        IQ_CV_ADB_STATUS="Not detected"
        _iq_cv_setup_info \
            "No ADB target detected. Connect the IQ9 over USB-C, enable USB debugging, accept the host key, and re-source setup.sh if you want to use mAP or test mode."
        return 0
    fi
    IQ_CV_ADB_STATUS="OK"
    _iq_cv_setup_info "ADB target detected and authorized."
}

_iq_cv_setup_note_qai_hub_status() {
    if [[ ! -f "${HOME}/.qai_hub/client.ini" ]]; then
        IQ_CV_QAI_HUB_STATUS="Not configured yet"
        _iq_cv_setup_info \
            "QAI Hub authentication is not configured yet. Run: qai-hub configure --api_token <YOUR_QAI_HUB_API_KEY>"
        return 0
    fi
    if ! qai-hub list-devices >/dev/null 2>&1; then
        IQ_CV_QAI_HUB_STATUS="Check failed"
        _iq_cv_setup_info \
            "QAI Hub authentication check failed. Authenticate after setup with: qai-hub configure --api_token <YOUR_QAI_HUB_API_KEY>"
        return 0
    fi
    IQ_CV_QAI_HUB_STATUS="OK"
}

_iq_cv_setup_activate_venv() {
    if [[ ! -d "${IQ_CV_SCRIPT_DIR}/.venv" ]]; then
        _iq_cv_setup_info "Creating .venv with uv."
        if ! (
            cd "${IQ_CV_SCRIPT_DIR}" &&
            uv venv .venv
        ); then
            if [[ ! -d "${IQ_CV_SCRIPT_DIR}/.venv" ]]; then
                _iq_cv_setup_error "Failed to create .venv."
                return 1
            fi
            _iq_cv_setup_info ".venv already exists. Reusing it."
        fi
    fi

    # shellcheck source=/dev/null
    if ! source "${IQ_CV_SCRIPT_DIR}/.venv/bin/activate"; then
        _iq_cv_setup_error "Failed to activate .venv."
        return 1
    fi
}

_iq_cv_setup_print_summary() {
    local python_path
    python_path="$(python3 -c 'import sys; print(sys.executable)')"
    _iq_cv_setup_info "Python executable: ${python_path}"
    _iq_cv_setup_info "Virtual environment: ${VIRTUAL_ENV:-<not active>}"
    _iq_cv_setup_info "QAI Hub authentication: ${IQ_CV_QAI_HUB_STATUS}"
    _iq_cv_setup_info "ADB devices:"
    if command -v adb >/dev/null 2>&1; then
        adb devices
    else
        printf 'adb not available\n'
    fi
}

_iq_cv_setup_qai_hub_reason() {
    case "${IQ_CV_QAI_HUB_STATUS}" in
        "OK")
            return 1
            ;;
        "Check failed")
            printf 'qai-hub auth check failed'
            ;;
        *)
            printf 'qai-hub auth not configured'
            ;;
    esac
}

_iq_cv_setup_adb_reason() {
    if [[ "${IQ_CV_ADB_STATUS}" == "OK" ]]; then
        return 1
    fi
    printf 'adb not detected'
}

_iq_cv_setup_test_reason() {
    if [[ "${IQ_CV_ADB_STATUS}" == "OK" ]]; then
        return 1
    fi
    printf 'adb not detected (direct on device inference is possible without adb)'
}

_iq_cv_setup_print_mode_status() {
    local mode_name="$1"
    shift
    local reasons=("$@")
    local reason_text=""
    local reason

    for reason in "${reasons[@]}"; do
        if [[ -z "${reason}" ]]; then
            continue
        fi
        if [[ -n "${reason_text}" ]]; then
            reason_text="${reason_text}; "
        fi
        reason_text="${reason_text}${reason}"
    done

    if [[ -z "${reason_text}" ]]; then
        printf '%s %s\n' "${mode_name}" "✅"
        return 0
    fi

    printf '%s %s - %s\n' "${mode_name}" "❌" "${reason_text}"
}

_iq_cv_setup_print_supported_modes() {
    local qc_reason=""
    local adb_reason=""
    local test_reason=""

    qc_reason="$(_iq_cv_setup_qai_hub_reason || true)"
    adb_reason="$(_iq_cv_setup_adb_reason || true)"
    test_reason="$(_iq_cv_setup_test_reason || true)"

    printf 'Supported modes\n'
    _iq_cv_setup_print_mode_status "qc" "${qc_reason}"
    _iq_cv_setup_print_mode_status "mAP" "${adb_reason}"
    _iq_cv_setup_print_mode_status "test" "${test_reason}"
}

_iq_cv_setup_main() {
    export IQ_CV_SCRIPT_DIR
    IQ_CV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export PATH="${HOME}/.local/bin:${PATH}"

    _iq_cv_setup_install_uv || return 1
    _iq_cv_setup_activate_venv || return 1
    _iq_cv_setup_sync_host_deps || return 1
    _iq_cv_setup_note_qai_hub_status || return 1
    _iq_cv_setup_install_host_packages || return 1
    _iq_cv_setup_verify_adb_target || return 1
    _iq_cv_setup_print_summary || return 1
    _iq_cv_setup_print_supported_modes || return 1
}

_iq_cv_setup_main
return $?
