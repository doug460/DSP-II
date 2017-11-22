'''
Created on Oct 24, 2017

@author: dabrown

This is my attempt to create a Kalman filter based the paper by Fujimoto

http://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
'''

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pass

   # intial parameters
    n_iter = 50
    sz = (n_iter,) # size of array
    x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
    
    Q = 1e-5 # process variance
    
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    H = 1                   # sensor model
    
    R = 0.1**2 # estimate of measurement variance, change to see effect
    
    # intial guesses
    xhat[0] = 0.0
    P[0] = 1.0
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
    
        # measurement update
        K[k] = H * Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k] * H)
        P[k] = (1-H*K[k])*Pminus[k]
        
    plt.figure()
    plt.plot(z,'k+',label='noisy measurements')
    plt.plot(xhat,'b-',label='a posteri estimate')
    plt.axhline(x,color='g',label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage')
    
    plt.figure()
    valid_iter = range(1,n_iter) # Pminus not valid at step 0
    plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
    plt.title('Error vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('$(Voltage)^2$')
    plt.setp(plt.gca(),'ylim',[0,.01])
    plt.show()














