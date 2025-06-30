import logging
import pathlib
import os
import sys

TOP = pathlib.Path(os.getenv("TOP"))
LOGGER = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(format='%(levelname)s:%(message)s', force=True, level=logging.INFO)
