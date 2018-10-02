import numpy as np
import cwt

def _unpad(matrix, num):
    unpadded = matrix[:,num:len(matrix[0])-num]
    return unpadded

def _padded_cwt(params, dt, dj, s0, J,mother, padding_len):
    padded = np.concatenate([params,params,params])
    #padded = np.pad(params, padding_len, mode='edge')
    wavelet_matrix, scales, freqs, coi, fft, fftfreqs = cwt.cwt(padded, dt, dj, s0, J,mother)
    wavelet_matrix = _unpad(wavelet_matrix, len(params)) 
    return (wavelet_matrix, scales, freqs, coi, fft, fftfreqs)

def _scale_for_reconstruction(wavelet_matrix,scales, dj, dt):
    scaled = np.array(wavelet_matrix)
    # mexican Hat
    c = dj / (3.541 * 0.867)

    #Morlet(6)
    #c = dj / (0.776 * np.pi**(-0.25))
    # Morlet(3)
    #c = dj / (1.83 * np.pi**(-0.25))
    
    for i in range(0, len(scales)):
        scaled[i]*= c*np.sqrt(dt)/np.sqrt(scales[i])
     
    return scaled

def combine_scales(wavelet_matrix, slices):
    combined_scales = []

    for i in range(0, len(slices)):
        combined_scales.append(sum(wavelet_matrix[slices[i][0]:slices[i][1]]))
    return np.array(combined_scales)

def cwt_analysis(f0, num_scales=12, scale_distance=1.0):

    f0_mean = np.mean(f0)
    f0_mean_sub = f0-f0_mean
    
    # setup wavelet transform
    dt = 0.005  # frame length
    dj = scale_distance  # distance between scales in octaves
    s0 = 0.005 # first scale, here frame length 
    J =  num_scales #  number of scales

    mother = cwt.Mexican_hat()
    #mother = cwt.Morletl(3)
    
    wavelet_matrix, scales, freqs, coi, fft, fftfreqs = _padded_cwt(f0_mean_sub, dt, dj, s0, J,mother, 400)
    wavelet_matrix = _scale_for_reconstruction(np.real(wavelet_matrix), scales, dj, dt)

    return wavelet_matrix

def cwt_synthesis(wavelet_matrix, mean = 0):
    return sum(wavelet_matrix[:])+mean
