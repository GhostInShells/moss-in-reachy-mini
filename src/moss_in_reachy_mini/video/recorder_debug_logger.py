import logging
import os
from typing import Final

LOGGER_NAME: Final[str] = "moss_in_reachy_mini.video.recorder"


def get_video_recorder_logger(storage_dir: str) -> logging.Logger:
    """Create (or reuse) a dedicated logger writing only to recorder.debug.log.

    This logger is intentionally isolated (propagate=False) so it won't pollute
    console output controlled by Rich/Prompt.
    """

    os.makedirs(storage_dir, exist_ok=True)
    log_path = os.path.join(storage_dir, "recorder.debug.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers (e.g. when worker is re-created).
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == log_path:
            return logger

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s [%(threadName)s] %(message)s",
        )
    )
    logger.addHandler(file_handler)
    return logger
