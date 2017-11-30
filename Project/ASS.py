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

if __name__ == '__main__':
    pass

#     dirData = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/School/School 2017 Fall/DSP/Project/'
    dirData =  'F:/Documents/School/School 2017 Fall/DSP/Project/'
    dirOut = dirData + 'data/'
    
    baseFreq = 12000
    baseSegment = 256
    floorCo = 0
    
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
    
    
    # get fft
    SEGS = np.fft.fft(dataSegs)
    
    # do spectral subtraction
    SEGS_OUT = np.copy(SEGS)
    for indx, SEG in enumerate(SEGS):
        SEGS_OUT[indx] = SEG/np.max(np.abs(SEG))*(np.abs(SEG) - NOISE)
        
    segs_out = np.fft.ifft(SEGS_OUT)
        
    dataOut = np.reshape(segs_out, baseSegment * segments)
    
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
    
    out = 'ass_%d.txt' % (desiredSNR)
    file  = open(dirOut + out,'w')
    file.write(buf)
    file.close()
    
    # have to plot stuff... figures...
    fig = plt.figure()
    plt.plot(data_org)
    plt.title('Origonal Data')
    plt.xlabel('n')
    plt.ylabel('x_o[n]')
    plt.savefig(dirOut + 'org_data.png')
    
    fig = plt.figure()
    plt.plot(data)
    plt.title('Input DATA at %ddB SNR' % (desiredSNR))
    plt.xlabel('n')
    plt.ylabel('x[n]')
    out = dirOut + 'inData_%dsnr.png' % (desiredSNR)
    plt.savefig(out)
    
    fig = plt.figure()
    plt.plot(dataOut)
    plt.title('SS at %ddB SNR' % (desiredSNR))
    plt.xlabel('n')
    plt.ylabel('y[n]')
    out = dirOut + 'SS_%dsnr.png' % (desiredSNR)
    plt.savefig(out)
    
        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    