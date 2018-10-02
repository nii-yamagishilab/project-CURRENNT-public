import numpy as np
import scipy.interpolate
import scipy.stats

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        
def _cut_boundary_vals(params, num_vals):
    cutted = np.array(params)
    for i in range(num_vals, len(params)-num_vals):
        if params[i] <=0 and params[i+1] > 0:
            for j in range(i, i+num_vals):
                cutted[j] = 0.0
        if params[i] >0 and params[i+1] <= 0:
            for j in range(i-num_vals, i+1):
                cutted[j] = 0.0
    return cutted

def _interpolate_zeros(params, method='pchip', min_val = 0):
    # min_val = np.nanmin(params)
    voiced = np.array(params, float)        
    for i in range(0, len(voiced)):
        if voiced[i] == min_val:
            voiced[i] = np.nan
        
    if np.isnan(voiced[-1]):
        voiced[-1] = np.nanmin(voiced)
    if np.isnan(voiced[0]):
        voiced[0] = scipy.stats.nanmean(voiced)

    not_nan = np.logical_not(np.isnan(voiced))

    indices = np.arange(len(voiced))
    if method == 'spline':
        interp = scipy.interpolate.UnivariateSpline(indices[not_nan],voiced[not_nan],k=2,s=0)
        # return voiced parts intact
        smoothed = interp(indices)
        for i in range(0, len(smoothed)):
            if not np.isnan(voiced[i]) :
                    
                smoothed[i] = params[i]
        return smoothed
    elif method =='pchip':
        interp = scipy.interpolate.pchip(indices[not_nan], voiced[not_nan])
    else:
        interp = scipy.interpolate.interp1d(indices[not_nan], voiced[not_nan], method)
    return interp(indices)

def _smooth(params, win, type="HAMMING"):
    """
    gaussian type smoothing, convolution with hamming window
    """
    win = int(win+0.5)
    if win >= len(params)-1:
        win = len(params)-1
    if win % 2 == 0:
        win+=1

    s = np.r_[params[win-1:0:-1],params,params[-1:-win:-1]]

    if type=="HAMMING":
        w = np.hamming(win)
        #third = int(win/3)
        #w[:third] = 0
    else:
        w = np.ones(win)
        
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[(win/2):-(win/2)]
    
def _peak_smooth(params, max_iter, win,min_win=2,voicing=[]):
    smooth=np.array(params)
    
    orig_win = win
    #win_reduce_step = (orig_win+1) / float(max_iter)
    win_reduce =  np.exp(np.linspace(np.log(win),np.log(min_win), max_iter))

    std = np.std(params)
    TRACE = False
    if TRACE:
        pylab.plot(params, 'black')
    for i in range(0,max_iter):

        smooth = np.maximum(params,smooth)
        if TRACE:
            if i> 0 and i % 5 == 0:
                pass
                pylab.plot(smooth,'gray',linewidth=1)
                raw_input()

        if len(voicing) >0:
            smooth = _smooth(smooth,int(win+0.5))
            smooth[voicing>0] = params[voicing>0]
        else:
            smooth = _smooth(smooth,int(win+0.5),type='rectangle')

        win = win_reduce[i]
    
    if TRACE:
        pylab.plot(smooth,'red',linewidth=2)
        raw_input()

    return smooth
    
def remove_outliers(lf0, trace=False):
    if np.mean(lf0[lf0>0])>10:
        raise("logF0 expected")
    fixed = np.array(lf0)
   
    interp = _interpolate_zeros(fixed,'linear')

    # iterative outlier removal
    # 1. compare current contour estimate to a smoothed contour and remove deviates larger than threshold
    # 2. smooth current estimate with shorter window, thighten threshold
    # 3. goto 1.
    
    # In practice, first handles large scale octave jump type errors,
    # finally small scale 'errors' like consonant perturbation effects and
    # other irregularities in voicing boundaries
    #
    # if this appears to remove too many correct values, increase thresholds
    num_iter = 20
    max_win_len = 200
    min_win_len = 20
    max_threshold = 2.5
    min_threshold = 0.25
    
    if trace:
        import matplotlib
        import pylab
        pylab.rcParams['figure.figsize'] = 20, 5
        pylab.figure()
        pylab.title("outlier removal")

    _std = np.std(interp)

    win_len =  np.linspace(max_win_len,min_win_len, num_iter+1)
    outlier_threshold = np.linspace(_std*max_threshold, _std*min_threshold, num_iter+1)
    for i in range(0, num_iter):
        
        smooth_contour = _smooth(interp, win_len[i])
       
        low_limit = smooth_contour - outlier_threshold[i]
        hi_limit = smooth_contour + outlier_threshold[i]*1.5 # bit more careful upwards, not to cut emphases

        fixed[interp<low_limit] = 0
        fixed[interp>hi_limit] = 0

        if trace:
            pylab.clf()
            pylab.title("outlier removal %d" % i)
            pylab.ylim(3.5,7)
            pylab.plot((low_limit), 'gray',linestyle='--')
            pylab.plot((hi_limit), 'gray',linestyle='--')
            pylab.plot((interp),linewidth=3)
            pylab.plot(lf0)
            pylab.draw()
            
        interp = _interpolate_zeros(fixed,'linear')
   
    if trace:
        raw_input("press any key to continue")

    return fixed

def interpolate(f0, method="true_envelope"):
    if method == "linear":
        return _interpolate_zeros(f0,'linear')
    elif method == "pchip":
        return _interpolate_zeros(f0, 'pchip')

    elif method == 'true_envelope':
        interp = _interpolate_zeros(f0)
       
        _std = np.std(interp)
        _min = np.min(interp)
        low_limit = _smooth(interp, 200)-1.5*_std
        low_limit[low_limit< _min] = _min
        hi_limit = _smooth(interp, 100)+2.0*_std
        voicing = np.array(f0)
        constrained = np.array(f0)
        constrained = np.maximum(f0,low_limit)
        constrained = np.minimum(constrained,hi_limit)

        interp = _peak_smooth(constrained, 100, 20,voicing=voicing)
        # smooth voiced parts a bit too
        interp = _peak_smooth(interp, 5, 2) #,voicing=raw)
        return interp
    else:
        raise("no such interpolation method: %s", method)
