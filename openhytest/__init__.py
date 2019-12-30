"""
Initialize the library.
:copyright: 2020, openhytest developer
:license: MIT
"""

from .preprocessing import *
from .modelclasses import *

__version__ = "unknown"
try:
    from ._version import __version__
except ImportError:
    pass
