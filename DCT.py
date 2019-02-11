#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:46:54 2019

@author: kostya
"""
'''
    U = 1/2*np.array([  [np.sqrt(2)/2,       np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2],
                    [np.cos(np.pi/16),   np.cos(np.pi*3/16),  np.cos(np.pi*5/16),  np.cos(np.pi*7/16),  np.cos(np.pi*9/16),  np.cos(np.pi*11/16), np.cos(np.pi*13/16), np.cos(np.pi*15/16)],
                    [np.cos(np.pi*2/16), np.cos(np.pi*6/16),  np.cos(np.pi*10/16), np.cos(np.pi*14/16), np.cos(np.pi*18/16), np.cos(np.pi*22/16), np.cos(np.pi*26/16), np.cos(np.pi*30/16)],
                    [np.cos(np.pi*3/16), np.cos(np.pi*9/16),  np.cos(np.pi*15/16), np.cos(np.pi*21/16), np.cos(np.pi*27/16), np.cos(np.pi*33/16), np.cos(np.pi*39/16), np.cos(np.pi*45/16)],
                    [np.cos(np.pi*4/16), np.cos(np.pi*12/16), np.cos(np.pi*20/16), np.cos(np.pi*28/16), np.cos(np.pi*36/16), np.cos(np.pi*44/16), np.cos(np.pi*52/16), np.cos(np.pi*60/16)],
                    [np.cos(np.pi*5/16), np.cos(np.pi*15/16), np.cos(np.pi*25/16), np.cos(np.pi*35/16), np.cos(np.pi*45/16), np.cos(np.pi*55/16), np.cos(np.pi*65/16), np.cos(np.pi*75/16)],
                    [np.cos(np.pi*6/16), np.cos(np.pi*18/16), np.cos(np.pi*30/16), np.cos(np.pi*42/16), np.cos(np.pi*54/16), np.cos(np.pi*66/16), np.cos(np.pi*78/16), np.cos(np.pi*90/16)],
                    [np.cos(np.pi*7/16), np.cos(np.pi*21/16), np.cos(np.pi*35/16), np.cos(np.pi*49/16), np.cos(np.pi*63/16), np.cos(np.pi*77/16), np.cos(np.pi*91/16), np.cos(np.pi*105/16)]])
    Z = np.dot(U, np.dot(a, U.T))
'''
    
import numpy as np
#DCT tarnsformation:
def dct_2d(A):
    U = np.ones([8,8])
    U[0,:] = U[0,:]*np.sqrt(2)/4
    for k in range(1, 8):
        arg = np.pi*k*(1/16)
        a = np.cos(np.linspace(arg, 15*arg, 8))
        U[k,:] = 1/2*a
    return np.dot(U, np.dot(A, U.T))

#np.dot(U, np.dot(A, U.T))
U = 1/2*np.array([  [np.sqrt(2)/2,       np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2,        np.sqrt(2)/2],
                    [np.cos(np.pi/16),   np.cos(np.pi*3/16),  np.cos(np.pi*5/16),  np.cos(np.pi*7/16),  np.cos(np.pi*9/16),  np.cos(np.pi*11/16), np.cos(np.pi*13/16), np.cos(np.pi*15/16)],
                    [np.cos(np.pi*2/16), np.cos(np.pi*6/16),  np.cos(np.pi*10/16), np.cos(np.pi*14/16), np.cos(np.pi*18/16), np.cos(np.pi*22/16), np.cos(np.pi*26/16), np.cos(np.pi*30/16)],
                    [np.cos(np.pi*3/16), np.cos(np.pi*9/16),  np.cos(np.pi*15/16), np.cos(np.pi*21/16), np.cos(np.pi*27/16), np.cos(np.pi*33/16), np.cos(np.pi*39/16), np.cos(np.pi*45/16)],
                    [np.cos(np.pi*4/16), np.cos(np.pi*12/16), np.cos(np.pi*20/16), np.cos(np.pi*28/16), np.cos(np.pi*36/16), np.cos(np.pi*44/16), np.cos(np.pi*52/16), np.cos(np.pi*60/16)],
                    [np.cos(np.pi*5/16), np.cos(np.pi*15/16), np.cos(np.pi*25/16), np.cos(np.pi*35/16), np.cos(np.pi*45/16), np.cos(np.pi*55/16), np.cos(np.pi*65/16), np.cos(np.pi*75/16)],
                    [np.cos(np.pi*6/16), np.cos(np.pi*18/16), np.cos(np.pi*30/16), np.cos(np.pi*42/16), np.cos(np.pi*54/16), np.cos(np.pi*66/16), np.cos(np.pi*78/16), np.cos(np.pi*90/16)],
                    [np.cos(np.pi*7/16), np.cos(np.pi*21/16), np.cos(np.pi*35/16), np.cos(np.pi*49/16), np.cos(np.pi*63/16), np.cos(np.pi*77/16), np.cos(np.pi*91/16), np.cos(np.pi*105/16)]])

A = np.ones([8,8])
A = A*100
print(dct_2d(A).round(1))