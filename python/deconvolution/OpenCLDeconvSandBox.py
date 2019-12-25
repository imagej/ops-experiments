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
import ArrayFireUtility
import DeconUtility

libcl=OpenCLDeconvUtility.getArrayFire()
libaf=ArrayFireUtility.getArrayFire()

# open image and psf
imgName='/home/bnorthan/code/images/Bars-G10-P15-stack-cropped.tif'
psfName='/home/bnorthan/code/images/PSF-Bars-stack-cropped.tif'

img=io.imread(imgName)
psf=io.imread(psfName)

# get next pow2
extDims=DeconUtility.nextPow2(img.shape)

img=img.astype('float32')
psf=psf.astype('float64')
psf=psf/psf.sum();
psf=psf.astype('float32')

print(img.shape)
print(psf.shape)

plt.imshow(img.max(axis=0))

out=np.zeros(img.shape).astype('float32')
deconv=np.copy(img);

(img, padding)=DeconUtility.padNDImage(img, extDims, 'reflect')
(psf, padding)=DeconUtility.padNDImage(psf, extDims, 'constant')
#out2=img.copy()
outcl=np.zeros(img.shape).astype('float32')
outaf=np.zeros(img.shape).astype('float32')

shifted_psf = np.fft.ifftshift(psf)

#libaf.conv2(img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, outcl);
#libcl.conv(img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, outaf);

#plt.imshow(outaf.max(axis=0))
#plt.imshow(outcl.max(axis=0))

deconcl=img.copy()
deconaf=img.copy()

#lib.conv(img.shape[0], img.shape[1], img.shape[2], img, shifted_psf, out2);
start=time.time()
libaf.deconv(100,img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, deconaf, img);
end=time.time()
aftime=end-start
print('af time', aftime)

start=time.time()
libcl.deconv(100,img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, deconcl, img);
end=time.time()
cltime=end-start
print('cl time', cltime)


#matplotlib inline
fig, ax = plt.subplots(1,3)
fig.set_figwidth(15)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(deconaf.max(axis=0))
ax[1].set_title('deconaf')

ax[2].imshow(deconcl.max(axis=0))
ax[2].set_title('opencl result')

