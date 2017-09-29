'''
Created on Sep 28, 2017

@author: dabrown

for homework 5. problem 3
'''

import numpy as np
import matplotlib.pyplot as plt

saveDir = '/media/dabrown/BC5C17EB5C179F68/Users/imdou/My Documents/School/School 2017 Fall/DSP/HW5/'

if __name__ == '__main__':
    pass

    #----------------------- PART A -----------------------

    # first get array that is 1000, zero-mean, unit variannce Gaussian random sequence
    noise = np.random.normal(0,1,(1000))
    
    noise = noise - np.mean(noise)
    noise = noise/np.std(noise)
    
    print('mean is ', np.mean(noise))
    print('Standard Deviatoin is ', np.std(noise))
    
    # calculate the total energy
    energy = 0
    for value in noise:
        energy += value ** 2
    
    print('Noise total Energy is ', energy)
    
    #----------------------- PART B -----------------------
    
    # pass through filter y[n] = 0.9y[n-1] + x[n]
    y = []
    y.insert(0,noise[0])
    for indx in range(1, len(noise)):
        y.insert(indx + 1, y[indx-1] * 0.9 + noise[indx])
    
    # get total energy of y
    energy_y = 0
    for value in y:
        energy_y += value ** 2
    
    print('Y total energy is ', energy_y)
    
    
    # stem plot of first 25 samples
    plt.figure(1)
    plt.subplot(211)
    plt.stem(noise[0:25],linefmt='r--', markerfmt='v')
    plt.title('Noise')
    
    plt.subplot(212)
    plt.stem(y[0:25], linefmt='b-.', markerfmt='x')
    plt.title('y[n]')
    plt.savefig(saveDir + "partb.png")
    
    #----------------------- PART C -----------------------
    
    # first apply transform from 5.84a to noise
    Hn = [[13, 13, 13, 13],[17, 7, -7,-17],[13, -13, -13, 13],[7, -17, 17, -7]]
    noise_hn = []
    
    # apply transform
    for indx in range(0, int(len(noise)/4)):
        noise_hn = np.append(noise_hn, (np.dot(Hn,noise[int(indx*4):int(indx*4 + 4)])))
        
    # get energy for each fourth element
    energy0 = 0
    energy1 = 0
    energy2 = 0
    energy3 = 0
    for indx in range(0, int(len(noise_hn)/4)):
        energy0 += noise_hn[indx * 4]**2
        energy1 += noise_hn[indx * 4 + 1] ** 2
        energy2 += noise_hn[indx * 4 + 2] ** 2
        energy3 += noise_hn[indx * 4 + 3] ** 2
        
    energy0 = energy0 / 676
    energy1 = energy1 / 676
    energy2 = energy2 / 676
    energy3 = energy3 / 676
    
    print('\n\nENERGY FOR Hn ON NOISE')
    print('Energy 0 is ', energy0)
    print('Energy 1 is ', energy1)
    print('Energy 2 is ', energy2)
    print('Energy 3 is ', energy3)
    print('Total energy is ', energy0 + energy1 + energy2 + energy3)
    
    #----------------------- PART D -----------------------
    
    # first apply transform from 5.84a to noise
    Hn = [[13, 13, 13, 13],[17, 7, -7,-17],[13, -13, -13, 13],[7, -17, 17, -7]]
    y_hn = []
    
    # apply transform
    for indx in range(0, int(len(y)/4)):
        y_hn = np.append(y_hn, (np.dot(Hn,y[int(indx*4):int(indx*4 + 4)])))
        
    # get energy for each fourth element
    energy0 = 0
    energy1 = 0
    energy2 = 0
    energy3 = 0
    for indx in range(0, int(len(y_hn)/4)):
        energy0 += y_hn[indx * 4]**2
        energy1 += y_hn[indx * 4 + 1] ** 2
        energy2 += y_hn[indx * 4 + 2] ** 2
        energy3 += y_hn[indx * 4 + 3] ** 2
        
    energy0 = energy0 / 676
    energy1 = energy1 / 676
    energy2 = energy2 / 676
    energy3 = energy3 / 676
    
    print('\n\nENERGY FOR Hn ON Y SIGNAL')
    print('Energy 0 is ', energy0)
    print('Energy 1 is ', energy1)
    print('Energy 2 is ', energy2)
    print('Energy 3 is ', energy3)
    print('Total energy is ', energy0 + energy1 + energy2 + energy3)
    
    
    #----------------------- PART E -----------------------
    
    # first apply transform from 5.84a to noise
    Gn = [[1, 1, 1, 1],[2, 1, -1, -2],[1, -1, -1, 1],[1, -2, 2, -1]]
    noise_hn = []
    
    # apply transform
    for indx in range(0, int(len(noise)/4)):
        noise_hn = np.append(noise_hn, (np.dot(Gn,noise[int(indx*4):int(indx*4 + 4)])))
        
    # get energy for each fourth element
    energy0 = 0
    energy1 = 0
    energy2 = 0
    energy3 = 0
    for indx in range(0, int(len(noise_hn)/4)):
        energy0 += noise_hn[indx * 4] ** 2
        energy1 += noise_hn[indx * 4 + 1] ** 2
        energy2 += noise_hn[indx * 4 + 2] ** 2
        energy3 += noise_hn[indx * 4 + 3] ** 2
        
    energy0 = energy0 / 4
    energy1 = energy1 / 10
    energy2 = energy2 / 4
    energy3 = energy3 / 10
    
    print('\n\nENERGY FOR Gn ON NOISE')
    print('Energy 0 is ', energy0)
    print('Energy 1 is ', energy1)
    print('Energy 2 is ', energy2)
    print('Energy 3 is ', energy3)
    print('Total energy is ', energy0 + energy1 + energy2 + energy3)
    
    #----------------------- PART F -----------------------
    
    # first apply transform from 5.84a to noise
    Gn = [[1, 1, 1, 1],[2, 1, -1, -2],[1, -1, -1, 1],[1, -2, 2, -1]]
    y_hn = []
    
    # apply transform
    for indx in range(0, int(len(y)/4)):
        y_hn = np.append(y_hn, (np.dot(Gn,y[int(indx*4):int(indx*4 + 4)])))
        
    # get energy for each fourth element
    energy0 = 0
    energy1 = 0
    energy2 = 0
    energy3 = 0
    for indx in range(0, int(len(y_hn)/4)):
        energy0 += y_hn[indx * 4]**2
        energy1 += y_hn[indx * 4 + 1] ** 2
        energy2 += y_hn[indx * 4 + 2] ** 2
        energy3 += y_hn[indx * 4 + 3] ** 2
        
    energy0 = energy0 / 4
    energy1 = energy1 / 10
    energy2 = energy2 / 4
    energy3 = energy3 / 10
    
    print('\n\nENERGY FOR Gn ON Y SIGNAL')
    print('Energy 0 is ', energy0)
    print('Energy 1 is ', energy1)
    print('Energy 2 is ', energy2)
    print('Energy 3 is ', energy3)
    print('Total energy is ', energy0 + energy1 + energy2 + energy3)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
