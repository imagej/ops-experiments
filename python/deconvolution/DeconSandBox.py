#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:33:19 2019

@author: bnorthan
"""

import imageio
import DeconUtility
import OpenCLDeconvUtility
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time
import YacuDecuUtility
import itk

# open image and psf
imgName='/home/bnorthan/code/images/Bars-G10-P15-stack-cropped.tif'
psfName='/home/bnorthan/code/images/PSF-Bars-stack-cropped.tif'

img=io.imread(imgName)
psf=io.imread(psfName)

extDims=DeconUtility.nextPow2(img.shape)

(img, padding)=DeconUtility.padNDImage(img, extDims, 'reflect')
(psf, padding)=DeconUtility.padNDImage(psf, extDims, 'constant')

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
psf=psf/psf.sum();
shifted_psf = np.fft.ifftshift(psf)
result = np.copy(img);
normal=np.ones(img.shape).astype(np.float32)

lib=YacuDecuUtility.getYacuDecu()
print('GPU Memory is',lib.getTotalMem())

start=time.time()
lib.deconv_device(100, int(img.shape[0]), int(img.shape[1]), int(img.shape[2]), img, shifted_psf, result, normal)
end=time.time()
cudatime=end-start
print('time is',end-start)

libcl=OpenCLDeconvUtility.getArrayFire()
deconcl=img.copy()

start=time.time()
libcl.deconv(100,img.shape[2], img.shape[1], img.shape[0], img, shifted_psf, deconcl, img);
end=time.time()
cvtime=end-start
print('cl time', end-start)

#matplotlib inline
fig, ax = plt.subplots(1,3)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(result.max(axis=0))
ax[1].set_title('result')

ax[2].imshow(deconcl.max(axis=0))
ax[2].set_title('opencl result')
'''
imgitk = itk.image_view_from_array(img)   # Convert to ITK object
psfitk = itk.image_view_from_array(psf)  # Convert to ITK object

print('try with ITK')
start=time.time()
deconvolved = itk.richardson_lucy_deconvolution_image_filter(
    imgitk,
    kernel_image=psfitk,
    number_of_iterations=100
)
end=time.time()
print(end-start)

result_itk = itk.array_from_image(deconvolved)

#matplotlib inline
fig, ax = plt.subplots(1,2)
ax[0].imshow(img.max(axis=0))
ax[0].set_title('img')

ax[1].imshow(result_itk.max(axis=0))
ax[1].set_title('result itk')
'''
 



