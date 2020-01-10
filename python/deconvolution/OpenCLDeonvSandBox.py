#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 06:58:25 2019

@author: bnorthan
"""

import imageio
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time
import OpenCLDeconvUtility


lib=OpenCLDeconvUtility.getArrayFire()

'''
# open image and psf
imgName='/home/bnorthan/code/images/Bars-G10-P15-stack-cropped.tif'
psfName='/home/bnorthan/code/images/PSF-Bars-stack-cropped.tif'

img=io.imread(imgName)
psf=io.imread(psfName)

img=img.astype('float32')
psf=psf.astype('float32')
psf=psf/psf.sum();

print(img.shape)
print(psf.shape)

plt.imshow(img.max(axis=0))

out=np.zeros(img.shape).astype('float32')
out2=np.zeros(img.shape).astype('float32')
deconv=np.copy(img);

shifted_psf = np.fft.ifftshift(psf)

#lib.conv(img.shape[2], img.shape[1], img.shape[0], img, psf, out);
#plt.imshow(out.max(axis=0))

lib.conv2(img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, out2);
plt.imshow(out2.max(axis=0))

start=time.time()

lib.deconv(100, img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, deconv, out2);

end=time.time()
print(end-start)
#plt.imshow(decon.max(axis=0))
'''