##
# Miscellaneous helper functions
##

from numpy import *

def random_weight_matrix(m, n):

    epsilon = sqrt(6) / sqrt(m + n)
    A0 = random.uniform(low=-epsilon, high=epsilon, size=(m, n))

    assert(A0.shape == (m,n))
    return A0