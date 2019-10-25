#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:33:19 2019

@author: bnorthan
"""

import imageio
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time
import YacuDecuUtility
import itk

# open image and psf
imgName='/home/bnorthan/code/images/Bars-G10-P15-stack-cropped.tif'
psfName='/home/bnorthan/code/images/PSF-Bars-stack-cropped.tif'

outName1='/home/bnorthan/code/images/Bars-RL-1.tif'
outName100='/home/bnorthan/code/images/Bars-RL-100.tif'

img=io.imread(imgName)
psf=io.imread(psfName)

print(img.shape)
print(psf.shape)

#matplotlib inline
fig, ax = plt.subplots(1,2)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(psf.max(axis=0))
ax[1].set_title('psf')

img=img.astype(np.float32)
psf=psf.astype(np.float32)
shifted_psf = np.fft.ifftshift(psf)
result1 = np.copy(img);
result100 = np.copy(img);
normal=np.ones(img.shape).astype(np.float32)

lib=YacuDecuUtility.getYacuDecu()
print('GPU Memory is',lib.getTotalMem())

lib.deconv_device(1, int(img.shape[0]), int(img.shape[1]), int(img.shape[2]), img, shifted_psf, result1, normal)
lib.deconv_device(100, int(img.shape[0]), int(img.shape[1]), int(img.shape[2]), img, shifted_psf, result100, normal)

io.imsave(outName1, result1)
io.imsave(outName100, result100)