import numpy as np
import math

if __name__ == '__main__':
    pass

    # so basically get the DFT of each, element multiply, then idft
    # for part a
    g = np.array([3, 2, -2, 1, 0, 1])
    h = np.array([-5, -1, 3, -2, 4, 4])
    
    G = np.fft.fft(g)
    H = np.fft.fft(h)
    
    # all the imaginary parts are ~ 0
    gh = np.real(np.fft.ifft(G*H))
    
    print("Part A:")
    print(repr(gh))
    
    
    
    # FOR PART B
    x = np.array([3- 2j, 4 - 1j, -2 + 3j, 1j, 0])
    v = np.array([1 - 3j, -2 - 1j, 2 + 2j, 3, -2 + 4j])
    
    X = np.fft.fft(x)
    V = np.fft.fft(v)
    
    xv = np.fft.ifft(X*V)
    
    print("\nPart B:")
    print(repr(xv))



    # FOR PART C
    w = np.array([1, 0, -1, 0, 1])
    y = np.array([1, 3, 9, 27, 81])
        
    W = np.fft.fft(w)
    Y = np.fft.fft(y)
    
    # imaginary parts are ~ 0
    wy = np.real(np.fft.ifft(W*Y))
    print("\nPart C:")
    print(repr(wy))








