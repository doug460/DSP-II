import math
import cmath
import numpy as np


if __name__ == '__main__':
    pass

    H = [4.0, 17.19 + 1.46*1j, -9.0 + 3.46*1j, -9.0 + 5.0*1j, 1.0 + 24.25*1j, 6.8 - 5.46*1j, 6.0, 6.8 + 5.46*1j, 1.0 - 24.25*1j, -9.0 - 5.0*1j, -9.0 - 3.46*1j, 17.19 - 1.46*1j ]
    
    g = []
    for n in range(0,len(H)):
         
        sum = 0 + 0*1j
        for k in range(0,len(H)):
            sum += H[k] * cmath.exp((-k * (n - 5)) * -1j*2*math.pi/12)
          
        sum = sum/12  
        g.append(sum)
    
    # imaginary parts ~0 so just not displaying them
    print("g = ")
    print(repr(np.round(np.real(g))))
    