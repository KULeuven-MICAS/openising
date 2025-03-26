import logging
import sys
import pathlib
import os

TOP = pathlib.Path(os.getenv("TOP"))
LOGGER = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
