import numpy as np

if __name__ == '__main__':
    pass

    #################3 PART A: #############
    print("FOR PART A: ")
    x = np.array([-3, 2, 4, -6, 1, 2])
    h = np.array([2, -1, 3, -4, 5, 6])
    
    g = x + h*1j
    
    # get dft
    G = np.fft.fft(g)
    
    G_shift = []
    G_shift.append(G[0])
    for n in range(1, len(G)):
        G_shift.append( G[len(G) - n] )
    
    X = (G + np.conjugate(G_shift)) * 0.5
    print("X = ")
    print(repr(np.round(X, 4)))
    
    H = (G - np.conjugate(G_shift)) * 0.5
    print("H = ")
    print(repr(np.round(H,4)))
    
    
    ################ PART B #################################
    print("\n\nFOR PART B")
    x = np.array([5, -4, -2, 6, 1, 3])
    h = np.array([4, -5, 5, 1, -2, 3])
    
    g = x + h*1j
    
    # get dft
    G = np.fft.fft(g)
    
    G_shift = []
    G_shift.append(G[0])
    for n in range(1, len(G)):
        G_shift.append( G[len(G) - n] )
    
    X = (G + np.conjugate(G_shift)) * 0.5
    print("X = ")
    print(repr(np.round(X,4)))
    
    H = (G - np.conjugate(G_shift)) * 0.5
    print("H = ")
    print(repr(np.round(H,4)))
    