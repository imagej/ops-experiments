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

def getArrayFire():
    print('getArrayFire')
    # load library
    lib=CDLL('libopencldeconv.so', mode=RTLD_GLOBAL)
    
    array_3d_float = npct.ndpointer(dtype=np.float32, ndim=3 , flags='CONTIGUOUS')   
    array_2d_float = npct.ndpointer(dtype=np.float32, ndim=2 , flags='CONTIGUOUS')
    array_1d_float = npct.ndpointer(dtype=np.float32, ndim=1 , flags='CONTIGUOUS')
    
    #lib.arrayTest.argtypes = [c_int, array_1d_float];
    lib.conv.argtypes = [c_int, c_int, c_int, array_3d_float, array_3d_float, array_3d_float]
    lib.fft2d.argtypes = [c_int, c_int, array_2d_float, array_2d_float]
    lib.fftinv2d.argtypes = [c_int, c_int, array_2d_float, array_2d_float]
    lib.deconv.argtypes = [c_int, c_int, c_int, c_int, array_3d_float, array_3d_float, array_3d_float, array_3d_float]
    
    print('gotarrayfire!!')
    
    return lib
    
    