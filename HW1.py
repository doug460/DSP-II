'''
Created on Aug 29, 2017

@author: dabrown

This is just for HW1
'''

import numpy as np
import math 
import random

import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass

    # for problem #M2.8 in book
    x = [2,0,-1,6,-3,2,0]
    y = [8,2,-7,-3,0,1,1]
    w = [3,6,-1,2,6,6,1]
    
    print(np.convolve(x,x))
    
    print("\n\n")
    print(np.convolve(y,y))
    
    print("\n\n")
    print(np.convolve(w,w))
    
    print("\n\n")
    print(np.convolve(x,y))
    
    print("\n\n")
    print(np.convolve(x,w))
    
    
    # for problem #M2.9 
    # corrupting signal x
    x = [2,0,-1,6,-3,2,0,8,2,-7,-3,0,1,1,3,6,-1,2,6,6,1]
    x_noise = []
    for i in range(0, len(x)):
        x_noise.insert(i,x[i] +3* np.random.rand())
        

    x_conv = np.convolve(x,x_noise)
    
    
    plt.figure(1)
    plt.plot(range(-20, 21, 1),abs(x_conv))
    plt.xlabel('l')
    plt.ylabel('abs(r_xx)')
    plt.title('Autocorrelation with noisy signal')
    
    
    # for 2.10
    a1 = 0.6
    a2 = 0.8
    
    
    x1= []
    x2 = []
    for n in range(0,21,1):
        x1.insert(n,a1**n) 
        x2.insert(n,a2**n)
    
    plt.figure(2)
    plt.title('a = 0.6')
    plt.xlabel('l')
    plt.ylabel('x1')
    plt.plot(range(-20, 21, 1), np.convolve(x1,x1))
    
    plt.figure(3)
    plt.title('a = 0.8' )
    plt.ylabel('x2')
    plt.xlabel('l')
    plt.plot(range(-20, 21, 1), np.convolve(x2,x2))
    
    plt.show()
    
    
    
    
    
    
    