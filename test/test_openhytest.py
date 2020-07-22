#    Copyright (C) 2020 by
#    Nathan Dutler <nathan.dutler@unine.ch>
#    Philippe Renard <philippe.renard@unine.ch>
#    Bernard Brixel <bernard.brixel@erdw.ethz.ch>
#    All rights reserved.
#    MIT license.

"""
Test module for the openhytest modules
Execute with pytest : `pytest test_openhytest.py`
"""
import sys
sys.path.append("..")

import openhytest as ht
import numpy as np
import pytest
import pandas as pd
from scipy.special import expn as E1

from scipy.integrate import quad

"""
Utilities to construct the tests
"""

def float_eq(a, b, tolerance=1e-4):
    """
    Returns True if the difference between a and b is lower
    than tolerance.
    """
    return abs(a-b) < tolerance


"""
Test 
"""

def Theis():
    """ Returns the drawdown for a simple Theis solution.
        The drawdown contains 20 points calculated on logspace.       
    """   
    t = np.array([0.100000000000000, 0.183298071083244, 0.335981828628378, 0.615848211066026, 1.12883789168469, 2.06913808111479, 3.79269019073225, 6.95192796177561, 12.7427498570313, 23.3572146909012, 42.8133239871940, 78.4759970351462, 143.844988828766, 263.665089873036, 483.293023857175, 885.866790410082, 1623.77673918872, 2976.35144163132, 5455.59478116852, 10000])
    s = np.array([0.0124574589351349, 0.0613899310042083, 0.172045125013825, 0.346260700209489, 0.570025738397648, 0.826740166867314, 1.10350001339534, 1.39186911629846, 1.68678217919244, 1.98532968580265, 2.28587942441350, 2.58752732805235, 2.88977608832897	, 3.19235317144164, 3.49510952910147	, 3.79796373782958, 4.10087134387359	, 4.40380808541921, 4.70676072333130, 5.00972203401915])
    #s2 = 0.5 * E1(1, 0.25 / t )
    d = {'t': t, 's': s}
    return pd.DataFrame(data=d)
    

"""
Main tests
"""
def test_diff():
    df = Theis()
    prep = ht.preprocessing(df=df)
    dev = prep.ldiff()
    assert float_eq(np.mean(dev.s.iloc[-8:]), 0.4921)
    dev = prep.ldiffs(npoints=21)
    assert float_eq(np.mean(dev.s.iloc[-8:]), 0.4771)
    dev = prep.ldiffb()
    assert float_eq(np.mean(dev.s.iloc[-8:]), 0.4978)
    dev = prep.ldiffh()
    assert float_eq(np.mean(dev.s.iloc[-8:]), 0.4995)

def test_flowdim():
    df = Theis()
    prep = ht.preprocessing(df=df)
    dev = prep.flowdim()
    assert float_eq(np.mean(dev.s.iloc[-8:]), 2.5911)

    
    
    

