"""
Centralized logging utilities for DepressoSpeech.

Provides:
    - setup_logger(): dual console+file logging with rotation
    - rotate_logs(): enforces max N log files per prefix
"""

import sys
import logging
from pathlib import Path
from datetime import datetime


MAX_LOGS_PER_PREFIX = 5


def rotate_logs(log_dir: str | Path, prefix: str, max_files: int = MAX_LOGS_PER_PREFIX):
    """Keep only the newest `max_files` log files matching `prefix*`.

    Deletes oldest files and removes stale symlinks.
    """
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return

    # Find all matching log files (exclude symlinks)
    log_files = sorted(
        [f for f in log_dir.glob(f"{prefix}*.log") if not f.is_symlink()],
        key=lambda f: f.stat().st_mtime,
    )

    # Delete oldest files beyond the limit
    while len(log_files) > max_files:
        oldest = log_files.pop(0)
        oldest.unlink()

    # Clean up stale symlinks
    for symlink in log_dir.glob(f"{prefix}*latest*"):
        if symlink.is_symlink() and not symlink.resolve().exists():
            symlink.unlink()


def setup_logger(
    log_dir: str = "logs",
    prefix: str = "fusion_training",
    max_files: int = MAX_LOGS_PER_PREFIX,
) -> Path:
    """Configure dual-output logging: concise console + detailed file.

    Automatically rotates to keep max_files per prefix.
    Returns the path of the new log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Rotate old logs BEFORE creating new one
    rotate_logs(log_dir, prefix, max_files - 1)  # -1 to leave room for new file

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{prefix}_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # File handler — full detail
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-5s │ %(message)s", datefmt="%H:%M:%S"
    ))
    root.addHandler(fh)

    # Console handler — clean, minimal
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    # Symlink for latest
    latest = log_dir / f"{prefix}_latest.log"
    latest.unlink(missing_ok=True)
    latest.symlink_to(log_path.name)

    return log_path
