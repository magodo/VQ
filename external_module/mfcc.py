#-*- encoding:utf-8 -*-
# This file includes routines to get MFCC feature coefficients.
# Author: Zhaoting Weng 2014

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.curdir, os.path.pardir), os.path.pardir)))

import numpy as np
#from scipy.fftpack import dct
from my_dct import my_dct
from preprocess import signal_to_powerspec

def mfcc(signal, fs = 8000, wlen = 0.025, inc = 0.01, nfft = 512, nfilt = 26, numcep = 13, lowfreq = 0, highfreq = None, preemph = 0.9375,
         ceplifter = 22, endpoint = False, appendEnergy = False):
    """Compute MFCC features from audio signal.
    :param signal: raw audio signal.
    :param fs: sample rate, default is 8000 Hz
    :param wlen: length of every frame in seconds, default is 25 ms
    :param inc: the steps between successive windows in seconds, default is 10 ms
    :param nfft: the fft length to use, default is 512.
    :param nfilt: the amount of filter bands to be used, default is 26.
    :param numcep: the number of cepstrum to return, default is 13
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.9375.
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param endpoint: whether to apply end-point detection when doing frame job, default is False.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy, default is False.

    :returns: A numpy array of size fn * numcep. Each row holds one MFCC feature vector.
    """
    # power spectrum(only get the first nfft/2+1 number of power spectrum of each frame) (fn * (nfft/2 + 1))
    pspec = signal_to_powerspec(signal, fs, wlen, inc, nfft, preemph, endpoint = endpoint)[:,:nfft/2+1]
    # Get total energy of each frame(fn * 1)
    energy = pspec.sum(1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    # Get Mel filter bank(nfilt * (nfft/2 + 1))
    fbank = filterbank(nfilt, fs, nfft, lowfreq, highfreq)
    # Energy of each filter bank for each frame(fn * nfilt)
    fenergy = np.dot(pspec, fbank.T)
    # Log filter bank energy(fn * nfilt)
    fenergy = np.where(fenergy == 0, np.finfo(float).eps, fenergy)               # if fenergy is zero, we get problems with log
    log_fenrgy = np.log(fenergy)
    # DCT 2
    #feature = dct(log_fenrgy, type = 2, axis = 1, norm = 'ortho')[:,:numcep]
    feature = my_dct(log_fenrgy, axis = 1, norm = True)[:, :numcep]

    # Lift cepstral coefficients
    feature = lifter(feature, ceplifter)
    # Replace first cepstral coefficient of each frame with log of its total frame energy
    if appendEnergy:
        feature[:,0] = np.log(energy)

    return feature

def filterbank(nfilt, fs, nfft, lowfreq = 0, highfreq = None):
    """Compute matrix of Mel filterbank.

    :param nfilt: Amount of melbank filters.
    :param fs: Sample rate.
    :param nfft: FFT size.
    :param lowfreq: Lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: Highst band edge of mel filters. In Hz, default is fs/2.
    :returns: A numpy array of Mel filterbank with size nfilt * (nfft/2 + 1). Each row holds one filter.
    """
    highfreq = highfreq or fs / 2

    lowmel = freq2mel(lowfreq)
    highmel = freq2mel(highfreq)
    # compute the points evenly spaced in mels
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    # convert from Mel to Hz to fft bin.
    bins = np.floor((nfft+1) * mel2freq(melpoints) / fs)
    bins = [int(i) for i in bins]

    filters = np.zeros([nfilt, nfft/2+1])
    for i in range(0, nfilt):
        for j in range(bins[i], bins[i+1]):
            filters[i,j] = (j - bins[i]) / float(bins[i+1] - bins[i])
        for j in range(bins[i+1], bins[i+2]+1):
            filters[i,j] = (bins[i+2] - j) / float(bins[i+2] - bins[i+1])
    return filters

def freq2mel(frequency):
    """Convert from frequency to Mel scale.

    :param frequency: Frequency to be converted.
    :returns: Corresponding Mel scale.
    """
    return 1125 * np.log(1 + frequency/700.0)

def mel2freq(mel):
    """Convert from Mel scale to frequency.

    :param mel: Mel scale to be converted.
    :returns: Corresponding frequency.
    """
    return 700.0 * (np.exp(mel/1125.0) - 1)

def lifter(cepstra,L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1+ (L/2)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

########################################
#               Main                   #
########################################
if __name__ == "__main__":
    import waveio

    # Get signal from .wav file
    signal, FS = waveio.wave_from_file("/home/magodo/code/voiceMaterial/4.wav")
    # MFCC
    feature = mfcc(signal, FS, endpoint = True)
    print feature
