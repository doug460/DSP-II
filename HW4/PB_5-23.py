import cmath
import numpy as np

if __name__ == '__main__':
    pass

    P = [3.5, -0.5 - 9.5*1j, 2.5, -0.5 + 9.5*1j]
    D = [17, 7.4 + 12*1j, 17.8, 7.4 - 12*1j]

    # imaginary parts are zero
    print("p = ", repr(np.real(np.fft.ifft(P))))
    print("d = ", repr(np.real(np.fft.ifft(D))))