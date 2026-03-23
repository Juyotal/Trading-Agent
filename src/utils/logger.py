import logging
import json
import sys
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        if hasattr(record, "trade_data"):
            log_entry["trade_data"] = record.trade_data
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{color}[{timestamp}] {record.levelname:8s}{self.RESET} | {record.module:20s} | {record.getMessage()}"


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("trading_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter())
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(log_path / f"trading_{date_str}.log")
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("trading_agent")
