#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:08:51 2019

@author: bnorthan
"""

from skimage import io
import numpy as np

# open image and psf
imgName='/home/bnorthan/Images/fromMM/ex6-2_CamB_ch0_CAM1_stack0198_488nm_8662163msec_0009953958msecAbs_000x_000y_000z_0000t.tif'
psfName='/home/bnorthan/Images/fromMM/PSF_488nm_dz100nm.tif'

img=io.imread(imgName).astype(np.float32)
psf=io.imread(psfName).astype(np.float32)

import matplotlib.pyplot as plt

