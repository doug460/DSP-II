'''
Created on Nov 29, 2017

@author: DougBrownWin
'''

from scipy.io.wavfile import read
from scipy.stats import signaltonoise
from scipy.signal import decimate
from scipy.io.wavfile import write
import numpy as np
from numpy import*
import matplotlib.pyplot as plt
from random import gauss
import cmath
import time


def power(signal):
    amp = np.sum(np.power(np.abs(signal),2))/len(signal)
    return amp

def getAlpha(Xe):
    if(Xe < 40):
        alpha = 2.5 - 2.5 * Xe / 40
    else:
        alpha = 0
        
    return alpha

def getInitial(SEG, NOISE, Xe, floorCo):
    alpha = getAlpha(Xe)
    
    Px = np.abs(SEG)
    Pn = NOISE
    
    Ps = Px - alpha * Pn
    
    size = len(SEG)
    P = np.zeros((size,size))
    
    for indx in range(size):
        P[indx][indx] = max([Ps[indx],floorCo*Px[indx]])

    return np.matrix(P)


def getH(k, baseSegment):
    H = np.zeros((baseSegment), dtype = np.complex128)
    for indx in range(baseSegment):
        H[indx] = (1/baseSegment)*cmath.exp(1j * 2 * math.pi * k * indx / baseSegment)
        
    return np.matrix(H)


if __name__ == '__main__':
    pass
#     dirData = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/School/School 2017 Fall/DSP/Project/'
    dirData =  'F:/Documents/School/School 2017 Fall/DSP/Project/'
    dirOut = dirData + 'data/'
    
    baseFreq = 12000
    baseSegment = 256
    floorCo = 0.85
    
    # get data and cut of ends
    rate, data =read(dirData + 'dsp3.wav')   
    data = data[2000:len(data)-200000]
    
    
    # down sample data to same that was used by authors
    lengthOld = len(data)
    factor = math.ceil(rate/baseFreq)
    data = decimate(data, factor, zero_phase = True)
    lengthNew = len(data)    
    
    # add noise
    dataPower = power(data)  
     
    desiredSNR = 20
    upper = sqrt(dataPower * 10**-(desiredSNR / 10))
    noise0 = [gauss(0.0, upper) for i in range(len(data))]
    
    
    
    # break data into 256 segments
    segments = math.floor(lengthNew / baseSegment) 
    data_org = np.copy(data[0:segments * baseSegment]) 
    data = data + noise0  
    data = data[0:segments * baseSegment]
    dataSegs = np.reshape(data, (segments, baseSegment))
    
    # get noise from first part of speech
    noise = dataSegs[0:10]
    
    
    
    NOISE = np.fft.fft(noise)
    
    # time averaged noise
    NOISE = np.abs(NOISE)
    NOISE = np.average(NOISE, axis = 0)
    
    # get covariance of noise
    R = np.cov(np.average(np.abs(noise), axis = 0))
    
    
    # get fft
    SEGS = np.fft.fft(dataSegs)
    
    # do spectral subtraction
    SEGS_OUT = np.copy(SEGS)
    
    
    # step through each segment
    for indx, segment in enumerate(dataSegs):
        print('%.1f %%' % (100 * indx/segments))
        # get fft
        SEG = np.fft.fft(segment)
        
        # get power of signal
        powx = power(segment)
        pown = power(noise)
        pows = sqrt(powx) + sqrt(pown)
        chi = 10*log10(pows*pows/pown)
        if(chi < 0):
            chi = 0
        
        # get P estimation
        P_minus = getInitial(SEG, NOISE, chi, floorCo)
        
        # get initial x
        x_minus = np.matrix(np.zeros(len(SEG)), dtype = np.complex128)
        
        
        # do kalman filter 
        for k in range(baseSegment):
            
            # get H (1,256)
            H = getH(k, baseSegment)
            
            # this is different than getH i defined
            # this return complex conjugate transpose of matrix!!
            # (256,1)
            H_star = H.getH()
            

            # get K (256,1)
            K = (P_minus * H_star) / (H * P_minus * H_star + R)
            
            # get updated x (1,256)
            x_minus = x_minus + np.multiply(K.T , (segment[k] - H* x_minus.T))
            
            # get updated P (256,256)
            P_minus = P_minus - np.multiply((H * K) , P_minus)
            
            
        SEGS_OUT[indx] = x_minus
            
    dataOut = np.fft.ifft(SEGS_OUT)
    dataOut = np.reshape(dataOut, baseSegment*segments)
    plt.plot(dataOut)
    plt.title('SS + K  at %ddB SNR' % (desiredSNR))
    plt.ylabel('y[n]')
    plt.xlabel('n')
    out = dirOut + 'SS_K_%dsnr.png' % (desiredSNR)
    plt.savefig(out)
    
    
    # get error (scale automatically just because
    maxIn = np.max(data)
    maxOut = np.max(dataOut)
    scaled = dataOut * maxIn / maxOut
    error = sqrt(np.mean(np.power(data-scaled,2)))

    # save string of info
    buf = 'RMS error is %.1f%%\n' % (error)
    buf += 'baseFreq %d\n' % (baseFreq)
    buf += 'Segment Size %d\n' % (baseSegment)
    buf += 'Flooring Coeficient %d\n' % (floorCo)
    buf += 'SNR %d \n' % (desiredSNR)
    print(buf)
    
    out = 'ass_k_%d.txt' % (desiredSNR)
    file  = open(dirOut + out,'w')
    file.write(buf)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    