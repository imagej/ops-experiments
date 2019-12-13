#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 06:58:25 2019

@author: bnorthan
"""

from skimage import data, io
import matplotlib.pyplot as plt
import numpy as np
import OpenCLDeconvUtility

# get coins image
image= data.coins().astype(np.float32)

# crop to 256 by 256 (TODO write code to find the closest supported and/or fast size)
image = np.ascontiguousarray(image[0:256, 0:256])

# size of first dimension of FFT assuming Hermitian interleaved 
M0=int(256)/int(2)+1;
M0=int(M0)
print("M0 is: ",M0)
M1=256

# create an output before of size 2 * M0 by M1 (*2 because FFTs are complex) 
out = np.zeros([256, M0*2]).astype(np.float32)

# get the lib
lib=OpenCLDeconvUtility.getArrayFire()

# call forward FFT
lib.fft2d(256, 256, image, out)

# case as complex, take absolute value and save
fftcomplex=out.view(np.complex64)
fftabs=np.abs(fftcomplex)
io.imsave('/home/bnorthan/Images/fromMM/fftabs.tif',fftabs)

print('call backward')
# inverse FFT
back=np.zeros([256, 256]).astype(np.float32)
lib.fftinv2d(256, 256, out, back)

#matplotlib inline
fig, ax = plt.subplots(1,3)
ax[0].imshow(image)
ax[1].imshow(fftabs)
ax[2].imshow(back)

