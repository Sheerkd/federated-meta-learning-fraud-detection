"""Application logging; callers must never log raw transaction features."""

import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler


def configure_logging(directory: Path, level: str = "INFO") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    for handler in (logging.StreamHandler(), RotatingFileHandler(
        directory / "application.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
