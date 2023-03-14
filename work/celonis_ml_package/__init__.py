# Version must always be a string
__version__ = "1.2"

# Default logging settings
LOGGER_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"standard": {"format": "%(asctime)s - %(message)s"}},
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        }
    },
    "loggers": {"celonis_ml": {"handlers": ["default"], "level": "INFO", "propagate": False}},  # root logger
}
