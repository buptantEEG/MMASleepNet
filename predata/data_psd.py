import os
import numpy as np
import math
import scipy.io as sio
from scipy.fftpack import fft,ifft


def PSD(data):
    '''
    compute  PSD
    --------
    input:  data [n*m]          n electrodes, m time points
            stft_para.stftn     frequency domain sampling rate
            stft_para.fStart    start frequency of each frequency band
            stft_para.fEnd      end frequency of each frequency band
            stft_para.window    window length of each sample point(seconds)
            stft_para.fs        original frequency
    output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
    '''
    #initialize the parameters
    # print(data.shape)
    STFTN=128
    fStart=[0,5,8,14,30]
    fEnd=[5,8,14,30,50]
    fs=100#stft_para['fs']
    window=30#stft_para['window']

    fStartNum=np.zeros([len(fStart)],dtype=int)
    fEndNum=np.zeros([len(fEnd)],dtype=int)
    for i in range(0,len(fStart)):
        fStartNum[i]=int(fStart[i]/fs*STFTN)
        fEndNum[i]=int(fEnd[i]/fs*STFTN)

    #print(fStartNum[0],fEndNum[0])
    n=data.shape[0]
    m=data.shape[1]

    #print(m,n,l)
    psd = np.zeros([n,len(fStart)])
    #Hanning window
    Hlength=window*fs
    #Hwindow=hanning(Hlength)
    Hwindow= np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength+1)) for n in range(1,Hlength+1)])

    dataNow=data[0:n]
    for j in range(0,n):
        temp=dataNow[j]
        Hdata=temp*Hwindow
        FFTdata=fft(Hdata,STFTN)
        # print('len(FFTdata)',len(FFTdata))
        magFFTdata=abs(FFTdata[0:int(STFTN/2)])
        # print(len(magFFTdata))
        for p in range(0,len(fStart)):
            E = 0
            #E_log = 0
            for p0 in range(fStartNum[p]-1,fEndNum[p]):
                E=E+magFFTdata[p0]*magFFTdata[p0]
            #    E_log = E_log + log2(magFFTdata(p0)*magFFTdata(p0)+1)
            E = E/(fEndNum[p]-fStartNum[p]+1)
            psd[j][p] = E
            #de(j,i,p)=log2((1+E)^4)
    
    return psd

