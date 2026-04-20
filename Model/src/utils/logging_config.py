"""
[LAYER_START] Session 10: Centralized Logging Configuration
Sets up structured logging to both console and file.

All terminal-facing scripts (train, extract, predict, serve) call
setup_logging() once at startup to route every module's logs to:
    1. Console (stdout) — colored, human-readable
    2. Log file (logs/<task>.log) — full detail with timestamps

[LOGGING] Every important event uses structured prefixes:
    [TRAINING_PATH]    — training pipeline events
    [INFERENCE_PATH]   — inference pipeline events
    [DATA_FLOW]        — data loading / feature dimensions
    [VALIDATION_CHECK] — input validation warnings
    [TIMING]           — per-stage performance timers
    [API]              — REST API events
    [CHECKPOINT]       — model/artifact save/load events
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Structured format for file logs
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
)
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Shorter format for console
CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    task_name: str,
    log_dir: str = "logs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_file: Optional[str] = None,
) -> Path:
    """
    Configure centralized logging for a script entry point.

    Call this ONCE at the start of each script (train.py, predict.py, etc.).
    All loggers across all modules will be captured.

    Args:
        task_name: Name of the task (e.g., "training", "inference", "extraction").
                   Used as the default log filename: logs/{task_name}.log
        log_dir: Directory for log files. Created if missing.
        console_level: Logging level for console output.
        file_level: Logging level for file output (usually more verbose).
        log_file: Override log filename. If None, uses {task_name}.log

    Returns:
        Path to the log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = log_file or f"{task_name}_{timestamp}.log"
    log_path = log_dir / filename

    # Symlink for quick access to latest run
    latest_link = log_dir / f"{task_name}_latest.log"

    # Reset root logger (in case setup_logging is called multiple times)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # --- Console handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
    )
    root.addHandler(console_handler)

    # --- File handler ---
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    file_handler.setFormatter(
        logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT)
    )
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "matplotlib", "PIL", "httpcore", "httpx"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"[CHECKPOINT] Logging initialized: console={console_level}, "
        f"file={file_level}, path={log_path}"
    )

    # Update latest symlink so `tail -f logs/training_latest.log` always works
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_path.name)
    except OSError:
        pass  # symlinks may not be supported on all filesystems

    return log_path
