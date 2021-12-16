import numpy as np
#def block_average():

def get_variance(samples):
    L = len(samples)
    mean = np.mean(samples)
    return np.sum((samples - mean)**2) / L


def make_blocks(A):
    """ 
        Transforms A into a new dataset with the same mean 
        value but with a new variance. The new dataset is 
        less correlated than the previous one.
    """
    A_new = np.zeros(len(A)//2 -1)
    for i in range(len(A_new)):
        A_new[i] = (A[2*i] + A[2*i+1])/2
    return A_new



def C(A):
    var = get_variance(A)
    C = var/(len(A)-1)
    return C

def ensemble_variance(A_uncorr):
    """
        Takes an uncorrelated data set and returns an estimate of the
        variance in the original data set.
    """
    L = len(A_uncorr)
    var = get_variance(A_uncorr)

    c = C(A_uncorr)
    d = np.sqrt(2*var**2/(L-1)**3)
    # Two values are returned: c +/- d.
    if c + d > c - d:
        return c + d
    else:
        return c - d

def get_block_average(A):
    count = 0
    while len(A) > 5:
        A_new = make_blocks(A)
        A = A_new
        count += 1
    var = ensemble_variance(A)
    std = np.sqrt(var)
    return std



def mean(A):
    return np.sum(A)/len(A)

def P(tB, AB, A):
    """ This approcahces t_corr for large block lengths. """
    var = get_block_variance(AB, A)
    return tB*var/(mean(A**2)-mean(A)**2)

def get_block_variance(AB, A):
    n = len(AB)
    var = np.sum((AB - mean(A))**2)/n
    return var

def get_correlation_time(A, t):
    """ Assuming A is measured a equillibrium. """
    P_list = np.zeros_like(t)
    for i in range(len(t)):
        tb = t[i]
        AB = create_blocks(A, t, tb)
        P_list[i] = P(tb, AB, A)
    return P_list

def time_average(A, tb):
    return np.sum(A)/tb

def create_blocks(A, t, tb):
    n = int(t[-1]/tb)
    AB = np.zeros(n)
    for i in range(n):
        AB[i] = time_average(A[:i], t[i])
    return AB


def get_var_from_correlation_time(A, tc, tB):
    var = tc*(mean(A**2)-mean(A)**2)/tB
    return var

