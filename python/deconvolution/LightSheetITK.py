#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:40:12 2019

@author: bnorthan
"""


from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import itk
import time
  

# open image and psf
imgName='/home/bnorthan/Images/fromMM/ex6-2_CamB_ch0_CAM1_stack0198_488nm_8662163msec_0009953958msecAbs_000x_000y_000z_0000t.tif'
psfName='/home/bnorthan/Images/fromMM/PSF_488nm_dz100nm.tif'

img=io.imread(imgName).astype(np.float32)
# crop image so it fits on the GPU
img=np.ascontiguousarray(img[:,256:-256,128:-128]);
print(img.shape)
psf=io.imread(psfName).astype(np.float32)
print(psf.shape)

print(psf.sum())
psf=psf/(psf.sum())
print(psf.sum())

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

io.imsave('/home/bnorthan/Images/fromMM/result_itk.tif',result_itk)