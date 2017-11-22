'''
Created on Nov 21, 2017

@author: dabrown

The paper this is based on is a pain in the Adaptive Spectral Subtraction :P
'''

from scipy.io.wavfile import read
from scipy.stats import signaltonoise
from scipy.signal import decimate
import numpy as np
from numpy import*
import matplotlib.pyplot as plt
from random import gauss
import cmath
import time



from wavTest import power

def rms(array):
    sum = 0
    for value in array:
        sum += value * value.conjugate()
        
    sum *= 1/(len(array))
    
    sum = sqrt(sum)
    
    return(sum)

def getAlpha(Xe):
    if(Xe < 40):
        alpha = 2.5 - 2.5 * Xe / 40
    else:
        alpha = 0
        
    return alpha

def getInitial(SEG, NOISE, Xe, floorCo):
    alpha = getAlpha(Xe)
    
    Px = np.power(np.abs(SEG), 2)/len(SEG)
    Pn = np.power(np.abs(NOISE), 2)/len(NOISE)
    
    Ps = Px - alpha * Pn
    
    size = len(SEG)
    P = np.zeros((size,size), dtype=np.complex128)
    
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

    dirData = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/School/School 2017 Fall/DSP/Project/'
    dirOut = dirData + 'data/'
    
    baseFreq = 6000
    baseSegment = 256
    floorCo = 1000
    
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
     
    data = data + noise0
    
    
    
     # break data into 256 segments
    segments = math.floor(lengthNew / baseSegment)    
    data = data[0:segments * baseSegment]
    dataSegs = np.reshape(data, (segments, baseSegment))

    # get noise from first part of speech
    noise = dataSegs[0]
    NOISE = np.fft.fft(noise)
    
    # get covariance of noise
    R = np.cov(noise)
    
    # array to hold output stuff
    SEG_OUT = np.zeros(dataSegs.shape, dtype = np.complex128)
    
    # how long it takes stuff to run
    start_time = time.time()

    # step through each segment
    for indx, segment in enumerate(dataSegs):
        print('%.1f %%' % (50 * indx/segments))
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
#         P_minus = np.zeros((256,256), dtype = np.complex128)
        
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
            
            
        SEG_OUT[indx] = x_minus
        
    # step through each segment
    for indx, segment in enumerate(dataSegs):
        print('%.1f %%' % (50 + 50 * indx/segments))
        # get fft
        SEG = SEG_OUT[indx]
        
        # get power of signal
        powx = power(np.abs(SEG))
        pown = power(noise)
        pows = sqrt(powx) + sqrt(pown)
        chi = 10*log10(pows*pows/pown)
        if(chi < 0):
            chi = 0
        
        # get P estimation
        P_minus = getInitial(SEG, NOISE, chi, floorCo)
#         P_minus = np.zeros((256,256),dtype = np.complex128)
        
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
            
            
        SEG_OUT[indx] = x_minus
    
    # plot origonal data
    fig = plt.figure()
    plt.title('Input Data %d dB SNR' % (desiredSNR))
    plt.xlabel('n')
    plt.ylabel('y')
    plt.plot(data)
    str = '%d_snr_in.png' % (desiredSNR)
    plt.savefig(dirOut + str)
    
    
    # get text stuff and save to txt file
    execTime = time.time() - start_time
    buf = 'execution time %d\n' % (execTime)
    buf += 'baseFreq %d\n' % (baseFreq)
    buf += 'Segment Size %d\n' % (baseSegment)
    buf += 'Flooring Coeficient %d\n' % (floorCo)
    buf += 'SNR %d \n' % (desiredSNR)
    
    str = '%d_snr.txt' % (desiredSNR)
    file  = open(dirOut + str,'w')
    file.write(buf)
    file.close()
   

    # get output data
    seg_out = np.fft.ifft(SEG_OUT)
    out = np.reshape(seg_out, (segments*baseSegment))
    
    plt.figure()
    plt.title('Filtered Data %d db SNR' % (desiredSNR))
    plt.ylabel('Estimated')
    plt.xlabel('n')
    plt.plot(out)
    str = '%d_snr_our.png' % (desiredSNR)
    plt.savefig(dirOut + str)
    
    plt.show()

    






















    
    
   
    
    
