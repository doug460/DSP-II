import numpy as np

if __name__ == '__main__':
    pass

    # example 5.13
    # so i believe this is just to do linear convoltion with dft
    g = np.array([1, 2, 0, 1, 0, 0, 0])
    h = np.array([2, 2, 1, 1, 0, 0, 0])
    
    G = np.fft.fft(g)
    H = np.fft.fft(h)
    
    gh = np.fft.ifft(H*G)
    
    print("Linear convolution")
    # imaginary parts are zero
    print(repr(np.real(gh)))