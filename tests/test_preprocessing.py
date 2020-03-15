#    Copyright (C) 2020 by
#    Nathan Dutler <nathan.dutler@unine.ch>
#    Philippe Renard <philippe.renard@unine.ch>
#    Bernard Brixel <bernard.brixel@erdw.ethz.ch>
#    All rights reserved.
#    MIT license.

"""
Test module for the prerpocessing
Execute with pytest : `pytest test_preprocessing.py`
"""

import openhytest as ht
import numpy as np
import pytest

def float_eq(a, b, tolerance=1e-6):
    """
    Returns True if the difference between a and b is lower
    than tolerance.
    """
    return abs(a-b) < tolerance