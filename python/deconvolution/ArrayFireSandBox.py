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
import ArrayFireUtility
import DeconUtility

# open image and psf
imgName='/home/bnorthan/code/images/Bars-G10-P15-stack-cropped.tif'
psfName='/home/bnorthan/code/images/PSF-Bars-stack-cropped.tif'

img=io.imread(imgName)

img=img+1;
psf=io.imread(psfName)

# get next pow2
extDims=DeconUtility.nextPow2(img.shape)

img=img.astype('float32')
psf=psf.astype('float32')
psf=psf/psf.sum();

print(img.shape)
print(psf.shape)

plt.imshow(img.max(axis=0))

(img, padding)=DeconUtility.padNDImage(img, extDims, 'reflect')
(psf, padding)=DeconUtility.padNDImage(psf, extDims, 'constant')
out2=np.zeros(img.shape).astype('float32')
deconv=np.copy(img);
shifted_psf = np.fft.ifftshift(psf)

lib=ArrayFireUtility.getArrayFire()

#lib.conv(img.shape[2], img.shape[1], img.shape[0], img, psf, out);
#plt.imshow(out.max(axis=0))
lib.conv2(img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, out2);
plt.imshow(out2.max(axis=0))

start=time.time()

lib.deconv(60, img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, deconv, deconv);

end=time.time()
print(end-start)
plt.imshow(deconv.max(axis=0))
