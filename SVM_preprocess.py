#Pre-processing inspired by Gabriel Garca. Only fourier used in final version
#https://github.com/gabrielgarza/exoplanet-deep-learning
import pandas as pd
import numpy as np
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from scipy.signal import butter

class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

##FUNDERA PÅ ORDNINGEN PÅ PROCESS-GREJSET
    def process(self, X, Xd):
        # Normalize
        if self.normalize:
            print("Normalizing...")
            X = normalize(X)
            Xd = normalize(Xd)

        if self.fourier:
            print("Applying Fourier...")
            X = np.abs(fft(X))
            Xd= np.abs(fft(Xd))

            # Keep first half of data as it is symmetrical after previous steps
            X = X[:,:(X.shape[1]//2)]
            Xd= Xd[:,:(Xd.shape[1]//2)]

        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            X = ndimage.filters.gaussian_filter1d(X, sigma=10) #--_filter1d eller inte?
            Xd = ndimage.filters.gaussian_filter1d(Xd, sigma=10)

        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            X = std_scaler.fit_transform(X)
            Xd = std_scaler.transform(Xd)
        
                # Apply fourier transform
        

        print("Finished Processing!")
        return X, Xd
