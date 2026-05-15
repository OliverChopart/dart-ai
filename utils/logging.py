"""Structured logging setup for the dart-ai system."""

import logging
import structlog


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging.

    Args:
        debug: If True, sets log level to DEBUG.
    """
    level = logging.DEBUG if debug else logging.INFO

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A structured logger instance.
    """
    return structlog.get_logger(name)
