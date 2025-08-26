import logging
from datetime import datetime


logger = logging.getLogger(__name__)


def _write_debug_log(message: str) -> None:
    """Write debug message to temporary file."""
    try:
        with open("/tmp/filewatcher_debug.log", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        logger.error("Failed to write debug log. ")
        pass
