from __future__ import absolute_import
from __future__ import print_function

import os
import scipy
import scipy.signal
import numpy as np
import scipy.io.wavfile
from   ioTools import readwrite

def spec(data, fft_bins=4096, frame_shift=40, frame_length=240):
    """
    f, t, cfft = scipy.signal.stft(data, nfft=4096, noverlap=frame_length-frame_shift, nperseg=frame_length)
    """
    f, t, cfft = scipy.signal.stft(data, nfft=4096, noverlap=frame_length-frame_shift, nperseg=frame_length)
    return f,t,cfft

def amplitude(cfft):
    """
    mag = amplitude(cfft)
    return spectral amplitude given FFT data (cfft) in complex numbers
    """
    mag = np.power(np.power(np.real(cfft),2) + np.power(np.imag(cfft),2), 0.5)
    return mag

def amplitude_re_im(data_re, data_im):
    """
    mag = amplitude_re_im(data_re, data_im)
    return spectral amplitude given real and imaginary part
    """
    mag = np.power(data_re * data_re + data_im * data_im, 0.5)
    return mag

def amplitude_to_db(mag):
    """
    20*np.log10(mag+ np.finfo(np.float32).eps)
    """
    return 20*np.log10(mag+ np.finfo(np.float32).eps)

def spec_amplitude(data,fft_bins=4096, frame_shift=40, frame_length=240):
    """
    return spectral amplitude in db, given speech waveform data
    mag_db = spec_amplitude(data,fft_bins=4096, frame_shift=40, frame_length=240):
    """
    _, _, cfft = spec(data, fft_bins, frame_shift, frame_length)
    mag  = amplitude(cfft)
    return amplitude_to_db(mag)

def fft_amplitude_db(data, nfft=1024):
    amp = np.fft.fft(data, nfft)[0:nfft/2+1]
    return amplitude_to_db(amplitude(amp))

def filter_res(data):
    w, h = scipy.signal.freqz(data, worN=4096/2)
    mag = amplitude(h)
    return amplitude_to_db(mag)

def read_fft_data_currennt(path, fft_length):
    data = readwrite.read_raw_mat(path, (fft_length/2+1)*2)
    data_re = data[:, 0::2]
    data_im = data[:, 1::2]
    return data_re, data_im

if __name__ == "__main__":
    pass
