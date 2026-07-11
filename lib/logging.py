import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '{"ts":"%(asctime)s","svc":"%(name)s","lvl":"%(levelname)s","msg":"%(message)s"}'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
