import numpy as np
import scipy.io.wavfile
from   ioTools import readwrite as py_rw
import multiprocessing
import os

def wavformConvert(wavdata, bit=16, signed=True, quanLevel = 255.0):
    """ wavConverted = wavformConvert(wavdata, bit=16, signed=True, quanLevel = 255.0)
        Assume wavData is int type:
        step1. convert int wav -> float wav
        step2. convert linear scale wav -> mu-law wav
        returned wav is integer but as float number
    """
    if signed==True:
        wavdata = np.array(wavdata, dtype=np.float32) / np.power(2.0, bit-1)
    else:
        wavdata = np.array(wavdata, dtype=np.float32) / np.power(2.0, bit)

    wavtrans = np.sign(wavdata) * np.log(1.0 + quanLevel * np.abs(wavdata)) / np.log(1.0 + quanLevel)
    wavtrans = np.round((wavtrans + 1.0) * quanLevel / 2.0)
    return wavtrans


def wavformDeconvert(wavdata, quanLevel = 255.0):
    """ waveformDeconvert = wavformDeconvert(wavdata, quanLevel = 255.0)
        Assume wavdata input is int type ([0, quanLevel])
        return the deconverted waveform in float number
    """
    wavdata = wavdata * 2.0 / quanLevel - 1.0
    wavdata = np.sign(wavdata) * (1.0/quanLevel) * (np.power((1+quanLevel), np.abs(wavdata)) - 1.0)
    return wavdata


def raw2wav(rawFile, wavFile, quanLevel = 255.0, bit=16, samplingRate = 16000):
    """ raw2wav(rawFile, wavFile, quanLevel = 255.0, bit=16, samplingRate = 16000)
        convert quantized wav [0, quanLevel] into wav file
    """
    transData = py_rw.read_raw_mat(rawFile, 1)
    recoData  = wavformDeconvert(transData, quanLevel)
    # recover to 16bit range [-32768, +32767]
    recoData  = recoData * np.power(2.0, bit-1)
    recoData[recoData >= np.power(2.0, bit-1)] = np.power(2.0, bit-1)-1
    recoData[recoData < -1*np.power(2.0, bit-1)] = -1*np.power(2.0, bit-1)
    # write as signed 16bit PCM
    if bit == 16:
        recoData  = np.asarray(recoData, dtype=np.int16)
    elif bit == 32:
        recoData  = np.asarray(recoData, dtype=np.int32)
    else:
        print "Only be able to save wav in int16 and int32 type"
        print "Save to int16"
        recoData  = np.asarray(recoData, dtype=np.int16)
    scipy.io.wavfile.write(wavFile, samplingRate, recoData)

def waveReadToFloat(wavFileIn):
    """ sr, wavData = wavReadToFloat(wavFileIn)
    
    """
    sr, wavdata = scipy.io.wavfile.read(wavFileIn)
    if wavdata.dtype is np.dtype(np.int16):
        wavdata = np.array(wavdata, dtype=np.float32) / np.power(2.0, 16-1)
    elif wavdata.dtype is np.dtype(np.int32):
        wavdata = np.array(wavdata, dtype=np.float32) / np.power(2.0, 32-1)
    else:
        print "Only be able to save wav in int16 and int32 type"
    return sr, wavdata

def waveSaveFromFloat(waveData, wavFile, bit=16, sr=16000):
    # recover to 16bit range [-32768, +32767]
    recoData  = waveData * np.power(2.0, bit-1)
    recoData[recoData >= np.power(2.0, bit-1)] = np.power(2.0, bit-1)-1
    recoData[recoData < -1*np.power(2.0, bit-1)] = -1*np.power(2.0, bit-1)
    # write as signed 16bit PCM
    if bit == 16:
        recoData  = np.asarray(recoData, dtype=np.int16)
    elif bit == 32:
        recoData  = np.asarray(recoData, dtype=np.int32)
    else:
        print "Only be able to save wav in int16 and int32 type"
        print "Save to int16"
        recoData  = np.asarray(recoData, dtype=np.int16)
    scipy.io.wavfile.write(wavFile, sr, recoData)
    

if __name__ == "__main__":
    pass
    #transData = py_rw.read_raw_mat('./mu16k/ATR_Ximera_F009_AOZORAR_03372_T01.raw', 1)
    #recoData  = wavformDeconvert(transData)
    #scipy.io.wavfile.write('./temp2.wav', 16000, recoData)
