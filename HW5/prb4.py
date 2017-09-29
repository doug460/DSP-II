'''
Created on Sep 28, 2017

@author: dabrown
'''

import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass


    saveDir = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/School/School 2017 Fall/DSP/HW5/'
    
    #origonal sequence is
    N = 64
    x = np.random.normal(0,1,(64))
    
    #data will hold the transform data
    Xdht = []
    
    for k in range(0,N):
        sum = 0
        for n in range(0,N):
            angle = 2*math.pi * n * k / N
            sum += x[n] * (math.cos(angle) + math.sin(angle))
        
        Xdht.insert(k,sum)
    
    
    # hold inverse
    xdht_i = []
    
    # now do inverse transform
    for n in range(0,N):
        sum = 0
        for k in range(0,N):
            angle = 2*math.pi * n * k / N
            sum += Xdht[k] * (math.cos(angle) + math.sin(angle))
        
        xdht_i.insert(n, sum/N)
    
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5)
    plt.subplot(311)
    plt.stem(x)
    plt.title('x[n]')
    
    plt.subplot(312)
    plt.stem(Xdht, markerfmt='*')
    plt.title('X[k]-DHT')
    
    plt.subplot(313)
    plt.stem(xdht_i)
    plt.title('x[n]-iDHT')
    
    plt.savefig(saveDir + 'prb4.png')
    
    
    
    # get difference between data
    error = np.sum(np.square(np.subtract(x, xdht_i)))
    print('Final error is ', error)
    
    
    
    
    
    