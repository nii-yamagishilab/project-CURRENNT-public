#!/usr/bin/python

import numpy as np
from tqdm import tqdm

np.random.seed(123)

class SpeechProcessing(object):
    def __init__(self, fl=1200, fs=240, fftl=2048):
        self.fl = fl
        self.fs = fs
        self.fftl = fftl
        winpower = np.sqrt(np.sum(np.square(np.blackman(self.fl).astype(np.float32))))
        self.window = {'ana' : np.blackman(self.fl).astype(np.float32) / winpower,
                       'syn' : winpower / 0.42 / (self.fl / self.fs)}

    def _frame(self, X):
        X = np.concatenate([np.zeros(int(self.fl/2), np.float32), X, np.zeros(int(self.fl/2), np.float32)])
        F = np.vstack([X[i:i+self.fl] for i in range(0, len(X)-self.fl, self.fs)])
        return F

    def _overlapadd(self, F):
        X = np.zeros(self.fs * (len(F) - 1) + self.fl, np.float32)
        for i in range(len(F)):
            X[i*self.fs:i*self.fs+self.fl] += F[i]
        X = X[int(self.fl/2):int(self.fl/2+self.fs*len(F)-self.fs/2)]
        return X

    def _anawindow(self, F):
        W = F * self.window['ana']
        return W

    def _synwindow(self, W):
        F = W * self.window['syn']
        return F

    def _rfft(self, W):
        Y = np.fft.rfft(W, n=self.fftl).astype(np.complex64)
        return Y

    def _irfft(self, Y):
        W = np.fft.irfft(Y).astype(np.float32)[:,:self.fl]
        return W
    
    def _amplitude(self, Y):
        A = np.absolute(Y)
        return A

    def _phase(self, Y):
        P = np.angle(Y)
        return P

    def _polar_to_rectangular(self, A, P):
        Y = A * np.exp(P * 1.0j)
        return Y

    def _griffinLim(self, A, N=300):
        P = np.pi * np.random.rand(A.shape[0], A.shape[1])
        for _ in tqdm(range(N)):
            X = self._overlapadd(self._synwindow(self._irfft(self._polar_to_rectangular(A, P))))
            P = self._phase(self._rfft(self._anawindow(self._frame(X))))
        return X

    def analyze(self, X):
        A = self._amplitude(self._rfft(self._anawindow(self._frame(X))))
        return A

    def generate(self, A):
        X = self._griffinLim(A)
        return X
