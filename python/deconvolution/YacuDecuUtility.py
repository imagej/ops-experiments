#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:22:56 2019

@author: bnorthan

based on code from here

https://github.com/koschink/PyYacuDecu

"""

from ctypes import *
import numpy as np
import numpy.ctypeslib as npct

def getYacuDecu():
    print('getYacuDecu')
    # load library
    lib=CDLL('libYacuDecu.so', mode=RTLD_GLOBAL)
    
    array_3d_float = npct.ndpointer(dtype=np.float32, ndim=3 , flags='CONTIGUOUS')
    
    lib.deconv_device.argtypes = [c_int, c_int,c_int,c_int, array_3d_float, array_3d_float, array_3d_float, array_3d_float];
    lib.getTotalMem.restype=c_longlong
    
    print('gotYacuDecu')
    
    return lib
    
    