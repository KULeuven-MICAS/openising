import pathlib
import os
import logging

TOP = pathlib.Path(os.getenv("TOP", pathlib.Path(__file__).resolve().parents[2]))
LOGGER = logging.getLogger(__name__)
