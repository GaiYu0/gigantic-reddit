import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.sparse as sps

def generate(n, p, q, rs=None):
    """
    Parameters
    ----------
    n : int
        Number of nodes.
    p :
    q :
    """
    k = len(p)
    if sum(p) == 1:
        s = npr.choice(k, size=n, p=p)
    elif sum(p) == n:
        s = np.hstack([np.full(x, i) for i, x in enumerate(p)])
    sizes = [np.sum(s == i) for i in range(k)]
    sigma = np.hstack([np.full(size, i, dtype=np.int) for i, size in enumerate(sizes)])

    kwargs = {'dtype' : np.float, 'random_state' : rs, 'data_rvs' : np.ones}
    blocks = [[sps.random(sizes[i], sizes[j], q[i][j], **kwargs) \
               for j in range(k)] for i in range(k)]
    a = sps.vstack([sps.hstack(x) for x in blocks])
    A = sps.triu(a) + sps.triu(a, k=1).transpose()

    return A, sigma
