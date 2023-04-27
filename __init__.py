# -*- coding: utf-8 -*-

"""Top-level package for scGIN-dev."""

__author__ = "zhjning"
__email__ = "zhjning@hotmail.com"
__version__ = "0.1.0"

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging
from logging import NullHandler
from ._settings import set_verbosity, set_seed

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

# default to INFO level logging for the scvi package
set_verbosity(logging.INFO)
# this prevents double outputs
logger.propagate = False

__all__ = ["set_verbosity", "set_seed"]


