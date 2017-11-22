'''
Created on Nov 21, 2017

@author: dabrown

basically import wav file, down sample, add noise, and see results
'''

from scipy.io.wavfile import read
from scipy.stats import signaltonoise
from scipy.signal import decimate
import numpy as np
from numpy import*
import matplotlib.pyplot as plt
from random import gauss


def power(signal):
    amp = 0
    for i in range(len(signal)):
        amp += signal[i] * signal[i]
    
    amp *= 1/(len(signal))
    
    return(amp)

def snrDb(signal, noise):
    p1 = power(signal)
    p2 = power(noise)
    
    return 10 * log10(p1/p2)

if __name__ == '__main__':
    pass

    dirData = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/School/School 2017 Fall/DSP/Project/'
    
    # get data and cut of ends
    rate, data =read(dirData + 'dsp3.wav')   
    data = data[2000:len(data)-20000]
    
    # down sample data to same that was used by authors
    lengthOld = len(data)
    factor = math.ceil(rate/12000)
    data = decimate(data, factor, zero_phase = True)
    lenthNew = len(data)
    
    print('Downsampled by ', lengthOld/lenthNew, ' and needed atleast ', rate/12000)

    # info about signal
    print('SNR of origonal signal is ', signaltonoise(data)) 
    dataPower = power(data)   
    print('Power of signal is ', dataPower)
    
    
#     plt.figure
#     plt.plot(data)
#     plt.show()
    
    
    ###############################
    # add 20 db noise
    # create white noise series
    desiredSNR = 20
    upper = sqrt(dataPower * 10**-(desiredSNR / 10))
    noise20 = [gauss(0.0, upper) for i in range(len(data))]
    print('\n------------ 20 dB ------------')
    print('Power of 20db noise is ', power(noise20))
    
    data20 = data + noise20
    print('New snr is ', snrDb(data, noise20))

    fig = plt.figure()
    plt.plot(data20)
    
    ####################################
    # noise doing 0 db noise
    desiredSNR = 0
    upper = sqrt(dataPower * 10**-(desiredSNR / 10))
    noise0 = [gauss(0.0, upper) for i in range(len(data))]
    
    print('\n---------------- 0 dB --------------')
    print('Power of 0db noise is ', power(noise0))
    
    data0 = data + noise0
    
    # esimated noise power
    print('esimated power noise is ', power(data0[0:100]))
    print('New snr is ', snrDb(data, noise0))
    print('scipy ', signaltonoise(data0))
    
    
    
    fig = plt.figure()
    plt.plot(data0)
    plt.show()  
    
    













