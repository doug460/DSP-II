'''
Created on Sep 12, 2017

@author: dabrown
'''

# this is for hw3
# this is for M4.2!! not 4.2.. MMMMM4.2

# use EQ (4.109) in prob 4.11 and show y[n] converges to sqrt(alpha) as n goes infinity

import matplotlib.pyplot as plt
import math

import numpy as np
if __name__ == '__main__':
    pass

    # y[n] = 0.5(y[n-1] + x[n]/y[n-1])
    # x[n] = alpha * stepResponse
    # y[-1] = 1
    
    tot_range = 10
    
    alpha = 25
    
    # define x
    x = []
    x.append(0)
    for indx in range(0,tot_range):
        x.append(alpha)
        
    # solve for y values
    y = []
    y.append(1)
    for indx in range(1,tot_range + 1):
        value = y[indx - 1] + x[indx]/y[indx - 1]
        value = value * 0.5
        y.insert(indx, value)
        
    # plot data
    line_y, = plt.plot(range(0,tot_range + 1),y)
    
    # define x and y for showing squareroot of alpha
    liney = []
    linex = []
    for indx in range(0,tot_range + 1):
        liney.insert(indx, math.sqrt(alpha))
        linex.insert(indx, indx)
    
    # plot squareroot line
    line_root, = plt.plot(linex, liney, 'k--')
    
    # label stuff
    plt.xlabel('n')
    plt.ylabel('y[n]')
    plt.title(r'Problem M4.2 for $\alpha = %d$' % (alpha))
    
    plt.legend([line_y, line_root], ['y[n]', r'$\sqrt{\alpha}$'])    
    plt.savefig('F:/Documents/School/School 2017 Fall/DSP/HW3/M4_2.png')
    plt.show()
    

    
    