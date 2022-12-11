import logging

"""
Custome logging

DEFAULT_FORMAT
The given default format for the logging files

GeerateLogger
Returns a new logger of the given name and log file
"""


# Basic logging config
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(
    level=logging.NOTSET,
)

DEFAULT_FORMAT = "%(levelname)s :: %(funcName)s :: %(message)s"


def GenerateLogger(name: __name__, Log_File: str, format: str = DEFAULT_FORMAT):
    new_logger = logging.getLogger(name)
    handler = logging.FileHandler(filename=Log_File, mode="w")
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    new_logger.addHandler(handler)
    new_logger.propagate = False

    return new_logger
