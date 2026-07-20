import logging
import sys
from lib.request_id import get_request_id


class _RequestIdFilter(logging.Filter):
    """Injects the current request's ID into every log record. Reads from
    the contextvar set by lib/request_id.py's middleware, so existing
    log.info(...) calls need no changes to pick this up."""
    def filter(self, record):
        record.request_id = get_request_id()
        return True


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '{"ts":"%(asctime)s","svc":"%(name)s","req_id":"%(request_id)s",'
            '"lvl":"%(levelname)s","msg":"%(message)s"}'
        ))
        handler.addFilter(_RequestIdFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger