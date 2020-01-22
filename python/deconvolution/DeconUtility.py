#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 07:04:20 2019

@author: bnorthan
"""

from math import ceil, floor
import numpy as np

def subtractBackground(img, percentOfMean):
    back=percentOfMean*img.mean()
    img=img-back;
    img[img<0]=0;
    print('ok')
    return img;

def getPadSize(img, psf):
    paddedSize=[];
    for i in range(len(img.shape)):
        paddedSize.append(img.shape[i]+psf.shape[i])
        
    print(paddedSize)
    return paddedSize
    
def padNDImage(img, paddedSize, padMethod):
    
    padding = [(int(floor((paddedSize[i]-img.shape[i])/2)), int(ceil((paddedSize[i]-img.shape[i])/2))) for i in range(len(paddedSize))]
    return np.pad(img, padding, padMethod), padding

def nextPow2(dims):
    out=[]
    for d in range(len(dims)):
        nextpow2=1;
        pow=1;
        while nextpow2<dims[d]:
            nextpow2=2**(pow)
            pow=pow+1
        out.append(nextpow2)
    return out
            
                    
                
    
    
    
    
