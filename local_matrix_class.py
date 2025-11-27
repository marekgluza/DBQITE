import numpy as np
class local_matrix:
    def __init__(self, local_matrix, L):
        self.local_matrix = local_matrix
        self.L = L
    def at( self, pos_x ):        
        return np.kron( np.kron( np.eye( 2**(pos_x-1) ), self.local_matrix ), np.eye( 2**(self.L-pos_x) ) )

sigma_z = [[1,0],[0,-1]]
sigma_y = 1j*np.array([[0,-1],[1,0]])
sigma_x = [[0,1],[1,0]]
sigma_plus = [[0,1],[0,0]]
sigma_minus = [[0,0],[1,0]]

class X_class(local_matrix):
    def __init__(self, L):
        local_matrix.__init__(self, sigma_x, L)
class Y_class(local_matrix):
    def __init__(self, L):
        local_matrix.__init__(self, sigma_y, L)
class Z_class(local_matrix):
    def __init__(self, L):
        local_matrix.__init__(self, sigma_z, L)
class S_plus_class(local_matrix):
    def __init__(self, L):
        local_matrix.__init__(self, sigma_plus, L)
class S_minus_class(local_matrix):
    def __init__(self, L):
        local_matrix.__init__(self, sigma_minus, L)
