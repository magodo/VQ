#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################
# Author: Zhaoting Weng
# Created Time: Sat 28 Mar 2015 03:01:48 PM CST
# File Name: VQ.py
# Description:
#########################################################################

import os
import sys

sys.path.append(os.path.abspath("/home/magodo/code/python/Edict/working/edict/lib/speech"))

from mfcc import mfcc
import waveio
import lbg

import numpy as np


#~~~~~~~~~~~~~
# Train a VQ
#~~~~~~~~~~~~~

def vq_generator(dirname, M, return_clusters = False):

    # Collect all MFCC vectors from each wave file from a voice_material collection
    # (To define as a dict is just a workaround to simulate nonlocal in python3.x).
    feature_collection = {"feature": []}

    def calc_mfcc(arg, dirname, fnames):

        for fname in fnames:
            if fname.endswith(".wav"):
                signal, FS = waveio.wave_from_file(os.path.join(dirname, fname))
                feature = list(mfcc(signal, FS))
                print "File: %s generate %d number of MFCC"%(fname, len(feature))
                feature_collection["feature"] += feature

    os.path.walk(dirname, calc_mfcc, None)
    print "Final amount of MFCC: %d"%len(feature_collection["feature"])

    # Perform M LBG classification, generate VQ: mu
    print "Begin generate VQ..."
    mu, clusters = lbg.lbg(feature_collection["feature"], M)
    print "Finish generate VQ."

    if not return_clusters:
        return mu
    else:
        return mu, clusters

if __name__ == "__main__":

    #~~~~~~~~~~~~~
    # Simple demo
    #~~~~~~~~~~~~~

    ## Calculate MFCC feature for demo.wav
    #signal, FS = waveio.wave_from_file("demo.wav")
    #feature = mfcc(signal, FS)
    #print feature
    #
    ## Perform 7 LBG classification
    #mu, clusters = lbg.lbg(feature, 7)

    #~~~~~~~~~~~~~
    # Real world demo
    #~~~~~~~~~~~~~
    from time import time

    M = 128
    timer = time()
    # Train VQ
    mu, clusters = vq_generator(dirname = "/home/magodo/code/voiceMaterial/word", M = M, return_clusters=True)
    # Store the trained VQ(tuple of mu(list of 256 different vectors) and clusters(dict of 256 different vector sets))
    import pickle
    with open("VQ.pkl"+"-%d"%M, "wb") as f:
        pickle.dump((mu,clusters), f)

    train_time = (time() - timer) / 60.0
    print "Used %f minutes" % train_time


    # Perform the trained VQ to a demo.wav
    signal, FS = waveio.wave_from_file("demo.wav")
    feature = list(mfcc(signal, FS))
    vq_array = []
    for i in feature:
        vq_array.append(lbg.cluster_point(i, mu))
    print vq_array





