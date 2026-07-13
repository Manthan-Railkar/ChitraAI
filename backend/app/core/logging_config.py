import sys
from pathlib import Path
from loguru import logger
from app.core.config import settings

def setup_logging() -> None:
    """
    Configures Loguru logging with a console handler and a rotating file handler.
    LogLevel is dynamically configured based on settings.LOG_LEVEL.
    """
    # Remove default handlers
    logger.remove()

    log_level = settings.LOG_LEVEL.upper()

    # Create logs directory at backend/logs
    backend_dir = Path(__file__).resolve().parent.parent.parent
    log_dir = backend_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "backend.log"

    # Standard log format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Console logging
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        backtrace=True,
        diagnose=True,
    )

    # Daily rotating file logging with compression
    logger.add(
        str(log_file),
        format=log_format,
        level=log_level,
        rotation="00:00",  # daily rotation at midnight
        retention="30 days",  # keep logs for 30 days
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    logger.info("Logging initialized successfully.")
