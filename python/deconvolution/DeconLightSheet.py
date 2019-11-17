#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 16:08:51 2019

@author: bnorthan
"""

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import DeconUtility
import YacuDecuUtility

# open image and psf
imgName='/home/bnorthan/Images/fromMM/ex6-2_CamB_ch0_CAM1_stack0198_488nm_8662163msec_0009953958msecAbs_000x_000y_000z_0000t.tif'
psfName='/home/bnorthan/Images/fromMM/PSF_488nm_dz100nm.tif'

img=io.imread(imgName).astype(np.float32)
psf=io.imread(psfName).astype(np.float32)

# crop image so it fits on the GPU
img=img[:,256:-256,128:-128]
img.shape
paddedNormal=np.ones(img.shape).astype(np.float32)

#matplotlib inline
fig, ax = plt.subplots(1,2)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(psf.max(axis=0))
ax[1].set_title('psf')

print(psf.min())
print(psf.mean())
print(psf.sum())
psf=DeconUtility.subtractBackground(psf,2)
psf=psf/(psf.sum())
print(psf.min())
print(psf.mean())
print(psf.sum())

#matplotlib inline
fig, ax = plt.subplots(1,2)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(psf.max(axis=0))
ax[1].set_title('psf')

paddedSize = DeconUtility.getPadSize(img, psf)

# HACK 
paddedSize[0]=315;

paddedImg, padding = DeconUtility.padNDImage(img, paddedSize, "constant")
paddedPSF, padding = DeconUtility.padNDImage(psf, paddedSize, "constant")
paddedNormal, padding = DeconUtility.padNDImage(paddedNormal, paddedSize, "constant")

print(paddedImg.shape)
print(paddedPSF.shape)
print(paddedNormal.shape)

paddedPSF = np.fft.fftshift(paddedPSF)

lib=YacuDecuUtility.getYacuDecu()
print('GPU Memory is',lib.getTotalMem())

lib.conv_device(int(paddedImg.shape[0]), int(paddedImg.shape[1]), int(paddedImg.shape[2]), paddedNormal, paddedPSF, paddedNormal, 1)
lib.removeSmallValues(paddedNormal, int(paddedNormal.shape[0]*paddedNormal.shape[1]*paddedNormal.shape[2]))
fig, ax = plt.subplots(1,1)
plt.imshow(paddedNormal.max(axis=0))

result = np.copy(paddedImg);

lib.deconv_device(100, int(paddedImg.shape[0]), int(paddedImg.shape[1]), int(paddedImg.shape[2]), paddedImg, paddedPSF, result, paddedNormal)

# unpad
result=result[padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], padding[2][0]:-padding[2][1]]
print(result.shape)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(result.max(axis=0))
ax[1].set_title('result')

io.imsave('/home/bnorthan/Images/fromMM/crop_img.tif',img)
io.imsave('/home/bnorthan/Images/fromMM/normal.tif',paddedNormal)
io.imsave('/home/bnorthan/Images/fromMM/result.tif',result)