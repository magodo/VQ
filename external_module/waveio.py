# -*- coding: utf-8 -*-
# This file includes routines for recording speech signal from voice and .wav file
# Author: Zhaoting Weng 2014

import time
import pyaudio
import numpy as np
import wave
from ctypes import *
import os
import sys

#------------------------------------------
# Handler for ALSA configuration error log
#------------------------------------------
#ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
#def py_error_handler(filename, line, function, err, fmt):
#    #print "messages are yummy"
#    pass
#c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
#asound = cdll.LoadLibrary('libasound.so')
#asound.snd_lib_error_set_handler(c_error_handler)

#-----------------------------------------
# Functions
#-----------------------------------------
def keep_record(stopdict, channels = 1, rate = 8000):
    """Keep record speech signal from voice until RecordStopFlag is set.

    :param channel: record channels(single or stereo)
    :param rate: sample rate
    """
    sampwidth = 2   # 16bit
    chunk = 1024
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pa.get_format_from_width(sampwidth),
                     channels = channels,
                     rate = rate,
                     frames_per_buffer = chunk,
                     input = True)
    str_wave_data = ""

    while not stopdict['flag']:
        str_wave_data += stream.read(chunk)
    wave_data = np.fromstring(str_wave_data, dtype = np.short)

    stream.stop_stream()
    stream.close()
    pa.terminate
    return wave_data

def recorde_t(duration, channels = 1, rate = 8000):
    """Record t(s) speech signal from voice

    :param duration: the record duraion
    :param channel: record channels(single or stereo)
    :param rate: sample rate
    :returns: numerical speech signal.
    """
    # 获取音频数据（二进制）
    sampwidth = 2   # 16bit
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pa.get_format_from_width(sampwidth),
                     channels = channels,
                     rate = rate,
                     input = True)
    num_sample = duration * rate
    str_wave_data = stream.read(num_sample)
    # 将音频数据转换为数组（S16型）
    wave_data = np.fromstring(str_wave_data, dtype = np.short)

    stream.stop_stream()
    stream.close()
    pa.terminate()
    return wave_data

def recorder_to_file(duration, filename, channels = 1, rate = 8000):
    """Record speech signal from voice and write into a .wav file

    :param duration: the record duraion
    :param filename: the destination file's name
    :param channels: record channels(single or stereo)
    :param rate: sample rate
    :returns: 0 if exits successfully
    """

    # 获取音频数据（二进制）
    sampwidth = 2
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pa.get_format_from_width(sampwidth),
                     channels = channels,
                     rate = rate,
                     input = True)
    num_sample = duration * rate
    str_wave_data = stream.read(num_sample)

    # 将音频数据写入文件
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(rate)
    wf.writeframes(str_wave_data)
    wf.close()

    stream.stop_stream()
    stream.close()
    pa.terminate()
    return 0

def wave_from_file(filename):
    """Read speech signal from .wav file

    :param filename: the filename of source file
    :returns: Tuple of numpy array object with content of speech signal and sample rate.
              Type: np.short
              Size: depends on .wav file(channels * frames)
    """
    # 从wav文件中读取音频数据
    wf = wave.open(filename, 'rb')
    samplerate = wf.getframerate()      # 获得采样率
    frame_num = wf.getnframes()         # 获得帧数
    str_wave_data = wf.readframes(frame_num)
    wave_data = np.fromstring(str_wave_data, dtype = np.short)
    wf.close()
    return (wave_data, samplerate)

def echo(data, channels = 1, rate = 8000):
    """Echo speech signal from wave data.

    :param data: the wave data
    :param channel: wave channels(single or stereo)
    :param rate: sample rate
    :returns: None"""
    #sampwidth = 2   # 16bit
    pa = pyaudio.PyAudio()
    stream = pa.open(format = pyaudio.paInt16,
                     channels = channels,
                     rate = rate,
                     output = True)

    stream.write(data, len(data))
    stream.stop_stream()
    stream.close()
    pa.terminate()


if __name__ == "__main__":
    import threading
#    # 1. Test for function: recorder
#    print "1. Test recorder: "
#    data = recorde_t(2, rate = 8000)
#    # 2.  Echo data
#    print "2.  Echo"
#    echo(data)
#
#
#    # 3. Test for function: recorder_to_file
#    print "3. Test recorder_to_file:"
#    recorder_to_file(2, "demo.wav")
#
#    # 4. Test for function: wave_from_file
#    print "4. Test wave_from_file: "
#    data, fs = wave_from_file("demo.wav")
#    print "Signal: ", data[:100]
#    print "Length of signal: ", len(data)
#    print "FS: ", fs

    # 5.  Test for keep recording
    stopdict = {'flag': False}

    def sample():
        data = keep_record(stopdict)
        echo(data)
        print len(data)

    def stop():
        time.sleep(2)
        stopdict['flag'] = True

    def test():
        t1 = threading.Thread(target = sample)
        t2 = threading.Thread(target = stop)
        t1.start()
        t2.start()

    test()



