import pathlib
import os
import logging

TOP = pathlib.Path(os.getenv("TOP"))
LOGGER = logging.getLogger(__name__)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
