import cwt_utils, f0_processing

import numpy as np
import sys

PLOT = False
if PLOT:
    import matplotlib.pyplot as pylab 
    pylab.ion()

def cwtF0Ana(f0, numScales=12, f0InterMethod=0, scaleDistance=1.0):
    # operate on log-domain
    lf0 = np.array(f0)
    lf0[f0>0]=np.log(f0[f0>0])
    lf0[f0<=0] = 0

    # tries to remove octave jumps other irregularities in iterative fashion
    f0_fix = f0_processing.remove_outliers(lf0, trace=PLOT)

    if f0InterMethod==0:
        # experimental interpolation method producing saggy interpolations
        f0_interp_fix = f0_processing.interpolate(f0_fix,'true_envelope')
    else:
        # another good choice is cubic hermite spline, does not overshoot
        f0_interp_fix2 = f0_processing.interpolate(f0_fix,'pchip')
    
    # do continuous wavelet tranform , returns scale matrix
    scales = cwt_utils.cwt_analysis(f0_interp_fix, num_scales=numScales, scale_distance=scaleDistance)
   
    # cwt produces zero mean, have to add mean to reconstruct the original
    #orig_mean = np.mean(f0_interp_fix)
    #reconstructed = sum(scales[:])+orig_mean

    phon_scales = []
    phon_scales.append(sum(scales[0:2])) #phone
    phon_scales.append(sum(scales[2:4])) #syllable
    phon_scales.append(sum(scales[4:6])) #word
    phon_scales.append(sum(scales[6:8])) #phrase
    phon_scales.append(sum(scales[8:]))  #utterance and beyond
    phon_scales = np.array(phon_scales)

    # cwt produces zero mean, have to add mean to reconstruct the original
    orig_mean = np.mean(f0_interp_fix)
    # reconstructed = sum(phon_scales[:])+orig_mean
    return phon_scales, orig_mean
    


def main():
    import sys

    f0 = np.loadtxt(sys.argv[1]) 

    # operate on log-domain
    lf0 = np.array(f0)
    lf0[f0>0]=np.log(f0[f0>0])
    lf0[f0<=0] = 0

    # tries to remove octave jumps other irregularities in iterative fashion
    f0_fix = f0_processing.remove_outliers(lf0, trace=PLOT)

    # experimental interpolation method producing saggy interpolations
    f0_interp_fix = f0_processing.interpolate(f0_fix,'true_envelope')

    # another good choice is cubic hermite spline, does not overshoot
    f0_interp_fix2 = f0_processing.interpolate(f0_fix,'pchip')

    if PLOT:
        pylab.rcParams['figure.figsize'] = 20, 5
        pylab.figure()
        pylab.title('interpolation')
        pylab.ylabel("Hz")
        pylab.plot(np.exp(lf0), 'black',label="original",linewidth=1, alpha=0.5)
        pylab.plot(np.exp(f0_interp_fix2), 'red',label="cubic hermite spline",linewidth=2)
        pylab.plot(np.exp(f0_interp_fix), 'blue',label="TE-style",linewidth=2)
        pylab.legend()
        raw_input("press any key to continue")


    # do continuous wavelet tranform , returns scale matrix
    scales = cwt_utils.cwt_analysis(f0_interp_fix, num_scales=12, scale_distance=1.0)
   
    # cwt produces zero mean, have to add mean to reconstruct the original
    orig_mean = np.mean(f0_interp_fix)
    reconstructed = sum(scales[:])+orig_mean
    
    if PLOT:
        pylab.figure()
        pylab.title("Complete CWT analysis")
        pylab.ylabel("scale")
        pylab.xlabel("frame")
        pylab.ylim(-1,scales.shape[0])
        for i in range(0,len(scales)):
            _std = np.std(f0_interp_fix)
            pylab.plot(scales[i]/_std*1.5+i,'black')
        pylab.contourf(scales, 100,cmap='afmhot')
        raw_input("press any key to continue")
        
    # combine adjacent scales as in SSW paper,
    # -> less parameters
    # -> less correlation between scales
    # -> better correspondence with linguistic units

    phon_scales = []
    phon_scales.append(sum(scales[0:2])) #phone
    phon_scales.append(sum(scales[2:4])) #syllable
    phon_scales.append(sum(scales[4:6])) #word
    phon_scales.append(sum(scales[6:8])) #phrase
    phon_scales.append(sum(scales[8:]))  #utterance and beyond
    phon_scales = np.array(phon_scales)

    if PLOT:
        pylab.figure()
        pylab.title("Adjacent scales combined")
        pylab.ylabel("combined scale")
        
        pylab.xlabel("frame")
        pylab.ylim(-0.5,phon_scales.shape[0]-0.5)
        for i in range(0,len(phon_scales)):
            _std = np.std(f0_interp_fix)
            pylab.plot(phon_scales[i]/_std*0.5+i,'black')
        raw_input("press any key to continue")

    # cwt produces zero mean, have to add mean to reconstruct the original
    orig_mean = np.mean(f0_interp_fix)
    reconstructed = sum(phon_scales[:])+orig_mean
               
    if PLOT:
        pylab.figure()
        pylab.ylabel('logHz')
        pylab.xlabel('frame')
        pylab.title("CWT synthesis, sum of scales")
        pylab.plot(f0_interp_fix, label="original")
        pylab.plot(reconstructed, label="reconstructed")
        pylab.legend()
        raw_input("press any key to continue")
        
        pylab.figure()
        pylab.title("Example of CWT manipulations")
        pylab.plot(np.exp(f0_interp_fix), label="original")
        smoothed = np.exp(sum(phon_scales[2:])+orig_mean)
        pylab.plot(smoothed, label="smoothing by removing fast scales")
        phon_scales[2]*=1.5
        postfiltered = np.exp(sum(phon_scales[:])+orig_mean)
        pylab.plot(postfiltered, label="word level enhanced")
        pylab.legend()
        
    raw_input()

if __name__=="__main__":
    main()
