#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 07:04:20 2019

@author: bnorthan
"""

def subtractBackground(img, percentOfMean):
    back=percentOfMean*img.mean()
    img=img-back;
    img[img<0]=0;