# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import re
from abc import ABCMeta
from datetime import datetime
from logging import Logger
from typing import List, Union

from pydantic import BaseModel


class NoInitFactory(type):
    def __call__(cls, *args, **kwargs):
        raise TypeError("Cannot instantiate factory directly")


class Singleton(type):
    _instances: dict[type, type] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonABC(ABCMeta, Singleton):
    pass


class DateRange(BaseModel):
    """
    DateRange contains start and end datetime timestamps
    :datetime start: The start time of range
    :datetime end: The end time of range
    """

    start: datetime
    end: datetime

    @classmethod
    def from_dict(cls, data: dict) -> "DateRange":
        """Create a DateRange from a dictionary. Accepts datetime objects or ISO 8601 strings.

        Args:
            data (dict): Must contain 'start' and 'end' keys mapping to datetime or ISO str.

        Raises:
            ValueError: If required keys are missing or cannot be parsed into datetime objects.
        """
        if not isinstance(data, dict):
            raise ValueError("date_range must be a dict with 'start' and 'end' keys")
        if "start" not in data or "end" not in data:
            raise ValueError("date_range dict must contain 'start' and 'end'")
        start_val = data["start"]
        end_val = data["end"]

        def _coerce(v):
            if isinstance(v, datetime):
                return v
            if isinstance(v, str):
                # Try common formats: ISO first
                try:
                    return datetime.fromisoformat(v)
                except ValueError:
                    # Fallback: attempt without microseconds
                    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]
                    for fmt in fmts:
                        try:
                            return datetime.strptime(v, fmt)
                        except ValueError:
                            continue
                raise ValueError(f"Could not parse datetime string: {v}")
            raise ValueError(f"Unsupported datetime type: {type(v)}")

        start_dt = _coerce(start_val)
        end_dt = _coerce(end_val)
        if end_dt < start_dt:
            raise ValueError("'end' datetime must be greater than or equal to 'start'")
        return cls(start=start_dt, end=end_dt)


def convert_to_win_path(mixed_path, drive_letter="C"):
    # Replace all forward slashes with backslashes
    normalized_path = mixed_path.replace("/", "\\")

    # Handle root directory conversion
    if normalized_path.startswith("\\"):
        normalized_path = f"{drive_letter}:{normalized_path}"

    return normalized_path


def format_output_as_list(output: Union[str, bytes]) -> List[str]:
    """
    Formats the str output of a command execution into a list of whitespace and
    newline stripped lines.

    Args:
        output (Union[str, bytes]): The output to format.

    Returns:
        List[str]: The formatted output as a list of strings.
    """

    if not output:
        return []
    output = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
    return output.splitlines()


def set_logging_level(level: Union[int, str] = "INFO") -> None:
    """
    Set the logging level for the root logger. All subsequent calls
    to loggers will have this level.

    level (Union[int, str]): A valid logging level for the python logging module
    """
    level_name = logging.getLevelName(level)
    file_info = ""
    if level_name == logging.DEBUG:
        file_info = "[%(filename)s:%(lineno)d in function %(funcName)s]"
    logging.basicConfig(
        format=f"%(asctime)s,%(msecs)d %(levelname)-3s {file_info} %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=level_name,
    )


def filter_log_by_date_range(stdout: bytes, date_range: DateRange, logger: Logger) -> bytes:
    """
    Stream-filter log data by date range, robust to various log formats:
      - Full ISO-like: YYYY-MM-DD HH:MM:SS(.sss|.uuuuuu)
      - Android logcat style: MM-DD HH:MM:SS(.sss) (year omitted, inferred from date_range.start)
    Handles:
      - Very large logs (streaming line iteration, no full regex over entire buffer)
      - Multiple timestamps on a single line (concatenated output fragments)
      - Multi-line log entries (continuation lines without leading timestamp appended to last included entry)
    """
    if not stdout:
        return b""
    try:
        output = stdout.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - extremely unlikely
        logger.debug("Failed to decode stdout as UTF-8; returning empty")
        return b""

    assumed_year = date_range.start.year
    # Timestamp core patterns (yearful or logcat month-day)
    ts_pattern = re.compile(r"(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2}) \d{2}:\d{2}:\d{2}(?:\.\d{3,6})?")

    entries: list[str] = []
    entry_timestamps: list[datetime] = []  # parallel list for potential multi-line continuation logic
    last_entry_included = False

    # Streaming iteration without building a huge list of lines
    line_start = 0
    length = len(output)
    while line_start < length:
        nl_pos = output.find("\n", line_start)
        if nl_pos == -1:
            line = output[line_start:]
            line_start = length
        else:
            line = output[line_start:nl_pos]
            line_start = nl_pos + 1

        # Find all timestamps in this line
        matches = list(ts_pattern.finditer(line))
        if not matches:
            # Continuation of previous log entry if prior entry was included
            if last_entry_included and entries:
                entries[-1] = entries[-1] + "\n" + line
            continue

        # There are one or more timestamps in the same physical line
        for idx, m in enumerate(matches):
            ts_str = m.group(0)
            # Determine slice for this entry's payload
            # For the first timestamp on a line, preserve any leading whitespace
            entry_start = 0 if idx == 0 else m.start()
            payload_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            payload = line[m.end() : payload_end]
            # Parse timestamp
            has_year = ts_str[0:4].isdigit() and ts_str[4] == "-"
            try:
                if has_year:
                    try:
                        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                    except ValueError:
                        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                else:
                    ts_with_year = f"{assumed_year}-{ts_str}"
                    try:
                        dt = datetime.strptime(ts_with_year, "%Y-%m-%d %H:%M:%S.%f")
                    except ValueError:
                        dt = datetime.strptime(ts_with_year, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                logger.debug(f"Skipping unparsable timestamp '{ts_str}'")
                last_entry_included = False
                continue

            if dt < date_range.start or dt > date_range.end:
                last_entry_included = False
                continue

            entry_text = line[entry_start:payload_end].rstrip("\r")
            entries.append(entry_text)
            entry_timestamps.append(dt)
            last_entry_included = True

    if not entries:
        logger.debug(
            f"No log entries found in date range: {date_range.start.isoformat()} - {date_range.end.isoformat()}"
        )
        return b""

    return ("\n".join(entries) + "\n").encode("utf-8")


def filter_log_by_string(device_log: bytes | str, log_filter: str, logger: Logger) -> bytes:
    """Filter a device log by a regex or literal substring returning full matching lines.

    Args:
        device_log: Raw device log content (bytes or str).
        log_filter: Regex pattern (re.MULTILINE) used to match each line. If the pattern compiles,
            any line with at least one match is retained.
        logger: Logger for diagnostics.

    Returns:
        bytes: UTF-8 encoded joined lines (with exactly one trailing newline) or b'' if no matches.
    """
    if not device_log:
        logger.debug("filter_log_by_string received empty log buffer")
        return b""
    if isinstance(device_log, bytes):
        try:
            decoded = device_log.decode("utf-8", errors="replace")
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to decode log bytes: {e}")
            return b""
    else:
        decoded = device_log
    try:
        rx = re.compile(log_filter, flags=re.MULTILINE)
    except re.error as e:
        logger.error(f"Invalid log_filter regex '{log_filter}': {e}")
        return b""
    matched: list[str] = []
    for line in decoded.splitlines():
        if rx.search(line):
            matched.append(line.rstrip("\r"))
    if not matched:
        logger.debug(f"No log lines matched filter: {log_filter}")
        return b""
    return ("\n".join(matched) + "\n").encode("utf-8", errors="replace")


def format_device_log(log_content: bytes | str | None, *, max_lines: int | None = None) -> bytes:
    """Normalize device log output (single-spacing only).

    Operations:
      - Decode bytes (utf-8, replace errors)
      - Normalize CRLF/CR to LF
      - Strip trailing whitespace per line
      - Optionally truncate to first max_lines
      - Join with single newlines
      - Ensure exactly one trailing newline
    """
    if not log_content:
        return b""
    if isinstance(log_content, bytes):
        text = log_content.decode("utf-8", errors="replace")
    else:
        text = str(log_content)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[max_lines:]
    normalized = "\n".join(lines).rstrip() + "\n"
    return normalized.encode("utf-8", errors="replace")
