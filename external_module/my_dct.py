#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
# Author: Zhaoting Weng
# Created Time: Sat 21 Mar 2015 11:45:37 AM CST
# File Name: my_dct.py
# Description:
#########################################################################

from scipy.fftpack import dct
import numpy as np
import math

def _my_dct(array, norm=False):
    """ Commit 1-D dct-2 transform. """
    array = np.array(array)
    if not array.dtype in [np.float, np.float16, np.float32, np.float64]:
        raise TypeError("my_dct not support: " + str(type(array.dtype)))
    N = len(array)
    out = np.array([2 * sum([array[n] * math.cos(math.pi*k/(2.0*N)*(2*n+1)) for n in range(N)]) for k in range(N)])
    if norm:
        f0 = math.sqrt(1.0/(4*N))
        fi = math.sqrt(1.0/(2*N))
        out[0] = out[0] * f0
        out[1:] = [i * fi for i in out[1:]]
    return out


def my_dct(array, axis = -1, norm=False):
    array = np.array(array)
    if axis == -1:
        return _my_dct(array.reshape(1,array.size)[0], norm)
    if axis in [0, 1]:
        if axis == 0:
            array = array.T
            for i in range(array.shape[0]):
                array[i] = _my_dct(array[i], norm)
            return array.T
        else:
            for i in range(array.shape[0]):
                array[i] = _my_dct(array[i], norm)
            return array
    else:
        raise TypeError("my_dct: axis is not in [-1, 0, 1]")

if __name__ == "__main__":

    #~~~~~~~~~~~~~~~~~
    # 1-D
    #~~~~~~~~~~~~~~~~~

    # scipy.fftpack.dct
    array = [4.0, 5.0, 6.0, 7.0, 2.0, 3.0, 4.0, 67.0, 234.0, 6.0, 457.0, 102.0]
    print "Result from dct: "
    print dct(array, type=2, axis=-1, norm="ortho")
    print "~~~~~~~~~~~~~~~~~~"

    # my dct (type 2)
    print "Result from my_dct: "
    print _my_dct(array, norm=True)
    print "~~~~~~~~~~~~~~~~~~"

    #~~~~~~~~~~~~~~~~~
    # 2-D
    #~~~~~~~~~~~~~~~~~

    array = [[4.0, 5.0, 6.0, 7.0, 2.0, 3.0], [4.0, 67.0, 234.0, 6.0, 457.0, 102.0]]
    print "Result from dct: "
    print dct(array, type=2, axis=1, norm="ortho")
    print "~~~~~~~~~~~~~~~~~~"

    print "Result from my_dct: "
    print my_dct(array, axis=1, norm=True)
    print "~~~~~~~~~~~~~~~~~~"

    print "Result from dct: "
    print dct(array, type=2, axis=0, norm="ortho")
    print "~~~~~~~~~~~~~~~~~~"

    print "Result from my_dct: "
    print my_dct(array, axis=0, norm=True)
    print "~~~~~~~~~~~~~~~~~~"

