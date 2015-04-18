# -*- encoding: utf-8 -*-
# This file includes routines for basic signal processing including framing, end-point detection and computing power spectrum
# Author: Zhaoting Weng 2014

import numpy as np
import math
#import scipy.signal as ss

def signal_to_powerspec(signal, fs = 8000, wlen = 0.025, inc = 0.01, nfft = 512, preemph = 0.9375, endpoint = False, show = False):
    """Convert from audio signal to power spectrum
    :param signal: raw audio signal(16-bit).
    :param fs: sample rate, default is 8000 Hz
    :param wlen: length of every frame in seconds, default is 25 ms
    :param inc: the steps between successive windows in seconds, default is 10 ms
    :param nfft: the fft length to use, default is 512.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.9375.
    :param endpoint: whether to apply end-point detection when doing frame job, default is False.
    :param show: whether plot frames based on end-point detection, defualt is False.

    :returns: A numpy array of size fn * nfft. Each row holds power spectrum of one frame.
    """
    signal = normalize(signal)
    signal = pre_emphasize(signal, preemph)
    frames = enframe(signal, wlen, inc, fs)

    if endpoint is True:
        # End point detection
        if show is True:
            pairs, amp, zcr, amp1, amp2, zcr2 = end_point(frames, wlen, inc, fs, verbose = True)             # Be verbose to plot

            # Choose only one speech segment if there is more or less than one segment
            frames, start_p, end_p= speech_segment_chooser(frames, pairs, verbose = True)                    # Be verbose to plot
            shower(signal, amp, zcr, wlen, inc, fs, start_p, end_p, amp1, amp2, zcr2)                        # Plot
            print "amp2: %d\n amp1: %d\n" %(amp2, amp1)
            framesize = frames.shape[0]                                                                      # Print more information
            print "Frame size: %d(frames)\nStart point: %d\nEnd point: %d\n" %(framesize, start_p, end_p)
        else:
            pairs = end_point(frames, wlen, inc, fs)
            # Choose only one speech segment if there is more or less than one segment
            frames = speech_segment_chooser(frames, pairs)

    # Convert frames to power spectrum
    return powspec(frames, nfft)

def normalize(signal):
    """Normalize 16-bit array.

    :param signal: 16-bit array.
    :returns: normalized float array.
    """
    max_number = max(abs(signal))
    return [i * 32767.0 / max_number for i in signal]

def pre_emphasize(signal, coeff = 0.9375):
    """Perform preemphasis on the input signal.

    :param signal: raw signal.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.9375.
    :returns: the filtered signal.
    """
    #return ss.lfilter([1, coeff], 1, signal)
    output = [0] * len(signal)
    output[1:] = [(signal[i] + coeff*signal[i-1]) for i in range(1, len(signal))]
    return output

def hamming(wlen):
    """ Generate hamming window with wlen length."""

    return np.array([(0.54-(0.46*math.cos(2*math.pi*i/(wlen-1)))) for i in range(wlen)])

def enframe(signal, wlen, inc, fs):
    """Frame a signal into overlapping frames.

    :param signal: raw signal.
    :param wlen: length of every frame in seconds, default is 25 ms
    :param inc: the steps between successive windows in seconds, default is 10 ms
    :param fs: sample rate.
    :returns: Framed and windowd numpy array. Size is fn * wlen
    """
    wlen = int(wlen * fs)
    inc = int(inc * fs)
    num_sample = len(signal)
    num_frame = int((num_sample - wlen) / inc + 1)
    #return np.array([signal[i*inc : i*inc+wlen] * ss.hamming(wlen) for i in range(num_frame)])   # scipy.signal.hamming() returns a numpy array
    return np.array([signal[i*inc : i*inc+wlen] * hamming(wlen) for i in range(num_frame)])

def zc_rate(frames):
    """Calculate zero cross rate of framed frames.

    :param frames: the numpy array of frames. Each row is a frame.
    :returns: zero-cross rate for every frame
    """
    fn = frames.shape[0]
    wlen = frames.shape[1]
    env_num = fn / 10                                                                   # The first "env_num" frames for estimate environment noise
    delta = np.mean(abs(frames[:env_num,:]))
    filterd = np.array([[i if abs(i) > delta else 0 for i in f] for f in frames])        # Carrier-amplitude regulation
    zcr = (filterd[:,:wlen-1] * filterd[:,1:wlen] < 0).sum(1)

    return zcr

def end_point(frames, wlen, inc, fs, verbose = False):
    """Perform end-point detection.

    :param frames: frames of signal.
    :param wlen: length of every frame in seconds
    :param inc: the steps between successive windows in seconds
    :param fs: sample rate.
    :param verbose: return verbose information(amp, zcr, amp1, amp1, zcr1) if true.

    :returns: If verbose is True, return tuple contains list of start/end point pairs and amp, zcr, amp1, amp2, zcr2;
              If verbose is False(default), return list of start/end point pairs.
    """

    fn = frames.shape[0]                    # Amount of frame
    wlen = frames.shape[1]                  # Length of one frame

    zcr = zc_rate(frames)                   # Short time zero-cross rate
    amp = np.square(frames).sum(1)          # Short time energy

    NIS = fn / 10                           # Number of frames to estimate the environment effects
    zcrth = np.average(zcr[:NIS])
    ampth = np.average(amp[:NIS])
    amp2 = 100 * ampth                        # Set threadshold
    amp1 = 400 * ampth
    zcr2 = 2 * zcrth

    # Get voice segments
    maxsilence = 5
    minlen = 8
    status = 0
    silence = 0
    count = 0
    start_p = []            # Reserve start point(frame) for every segment
    end_p = []              # Reserve end point(frame) for every segment
    for i in range(fn):
        if status == 0:
            # Maybe voice or no voice
            if amp[i] > amp1:
                # Voice
                status = 1
                count += 1
                silence = 0                         # init silence
                start_p.append(i - count + 1)               # record start point( Prepend the 'silence' segment)
            elif amp[i] > amp2 or zcr[i] > zcr2:
                # Maybe voice
                count += 1
            else:
                # No voice
                count = 0
        elif status == 1:
            # Voice segment
            if (amp[i] > amp2 and zcr[i] > zcr2) or (amp[i] > amp1):
                # Still voice
                count += 1
                silence = 0
            else:
                # No voice but maybe silence duration
                silence += 1
                if silence < maxsilence:
                    # Silence duration
                    count += 1
                else:
                    # Silence duration end
                    if count <= minlen:
                        # Noise segment
                        status = 0
                        count = 0
                        silence = 0
                        start_p.pop()
                    else:
                        # Voice segment end
                        status = 0
                        count = 0
                        silence = 0
                        end_p.append(i)
    # In case of voice is not end until frames wind up
    if status == 1 and count > minlen:
        end_p.append(fn)

    endpoints = zip(start_p, end_p)
    # Return Value
    if verbose is not False:
        return (endpoints, amp, zcr, amp1, amp2, zcr2)
    else:
        return endpoints

def speech_segment_chooser(frames, pairs, verbose = False):
    """Check amount of speech segment after end-point detection and choose only one segment.

    :param frames: numpy array of the enframed signal.
    :param pairs: list of start/end point returned by end-point detection.
    :param verbose: verbose mode, return start&end point of choosed speech segment if verbose mode is on.

    :returns: one speech segment frames.(numpy array), plus start&end point if verbose mode is on.
    """
    # Check if only one speech segment
    if len(pairs) is 0:
        print "Zero speech segment detected!!! Assume every frame is under ues!!!"
        pass
    elif len(pairs) > 1:
        print "More than one speech segment detected!!! Use the longest one as the efficient one!!!"
        span = [j - i for (i,j) in pairs]
        pair = pairs[span.index(max(span))]
        frames = frames[pair[0]:pair[1]]
    else:
        #print "Only one speech segment :>"
        pair = pairs[0]
        frames = frames[pair[0]:pair[1]]
    if verbose is True:
        return frames, pair[0], pair[1]
    else:
        return frames




def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the numpy array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the magnitude spectrum of the corresponding frame.
    """
    complex_spec = np.fft.rfft(frames,NFFT)
    return np.absolute(complex_spec)

def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the numpy array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0/NFFT * np.square(magspec(frames,NFFT))

def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.

    :param frames: the numpy array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 1.
    :returns: If frames is an NxD matrix, output will be NxNFFT. Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames,NFFT);
    ps[ps<=1e-30] = 1e-30
    lps = 10*np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps

def shower(raw, amp, zcr, wlen, inc, fs, start_p, end_p, amp1, amp2, zcr2):
    """ Show raw signal plot, short-time energy plot and short-time zero-cross rate plot.

    :param raw: speech signal.
    :param amp: array of short-time energy of every frame.
    :param zcr: array of short-time zero-cross rate of every frame.
    :param wlen: length of every frame in seconds
    :param inc: the steps between successive windows in seconds
    :param fs: sample rate.
    :param start_p: start frame point.
    :param end_p: end frame point.
    :param amp1: higher threshold of shor-time energy
    :param amp2: lower threshold of shor-time energy
    :param zcr2: lower threshold of shor-time zero-cross rate

    :returns: None.
    """
    import matplotlib.pyplot as plt

    fn = int((len(raw)- wlen) / inc + 1)         # Frame amount

    # convert segments from frame to time
    start_t = frame2time(start_p, inc)
    end_t = frame2time(end_p, inc)

    # Fig
    fig = plt.figure()

    # Axes.1
    ax1 = fig.add_subplot(311)
    t = [i/(fs*1.0) for i in range(len(raw))]
    ax1.plot(t, raw)
    ax1.axvline(x = start_t, ymin = -32768, ymax = 32767, c = 'r')
    ax1.axvline(x = end_t, ymin = -32768, ymax = 32767, c = 'r')

    ax1.grid(True)
    ax1.set_xlabel('Time/s')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Raw Signal')
    ax1.set_xlim(right = frame2time(end_p + 1, inc))

    # Axes.2
    ax2 = fig.add_subplot(312)
    x = range(len(amp))
    ax2.plot(x, amp)
    amp_limit = max(amp)
    ax2.axvline(x = start_p, ymin = -amp_limit, ymax = amp_limit, c = 'r')
    ax2.axvline(x = end_p, ymin = -amp_limit, ymax = amp_limit, c = 'r')
    ax2.axhline(y = amp1, xmin = 0, xmax = fn-1, c = 'b', ls = '--')
    ax2.axhline(y = amp2, xmin = 0, xmax = fn-1, c = 'b', ls = '--')

    ax2.grid(True)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Short-time Energy')
    ax2.set_xlim(right = end_p + 1)

    # Axes.3
    ax3 = fig.add_subplot(313)
    x = range(len(zcr))
    ax3.plot(x, zcr)
    zcr_limit = max(zcr)
    ax3.axvline(x = start_p, ymin = -zcr_limit, ymax = zcr_limit, c = 'r')
    ax3.axvline(x = end_p, ymin = -zcr_limit, ymax = zcr_limit, c = 'r')
    ax3.axhline(y = zcr2, xmin = 0, xmax = fn-1, c = 'b', ls = '--')

    ax3.grid(True)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Short-time Zero-cross rate')
    ax3.set_xlim(right = end_p + 1)

    # Plot
    plt.show()

def frame2time(frame, inc):
    """ Convert frame point to corresponding time.

    :param frame: frame point.
    :param inc: the steps between successive windows in seconds

    :returns: corresponding time.
    """
    return frame*inc



if __name__ == "__main__":
    import waveio

    # Get signal from .wav file
    raw, FS = waveio.wave_from_file("/home/magodo/code/voiceMaterial/7a.wav")
    # Define Consts
    WLEN = 256.0 / FS
    INC = 128.0 / FS
    pspec = signal_to_powerspec(raw, FS, WLEN, INC, endpoint = True, show = True)




