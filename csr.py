import numpy as np
import scipy.sparse as sps

sau = np.load('sau.npy')
sav = np.load('sav.npy')
m = len(sau)
n = max(np.max(sau), np.max(sav)) + 1
a2s = sps.coo_matrix((np.ones(m), (sau, sav)), shape=(n, n)).tocsr()
s2a = sps.coo_matrix((np.ones(m), (sav, sau)), shape=(n, n)).tocsr()
print('len(a2s.indptr)', len(a2s.indptr))
print('len(a2s.indices)', len(a2s.indices))
np.savetxt('a2s-indptr', a2s.indptr, '%d')
np.savetxt('a2s-indices', a2s.indices, '%d')
print('len(s2a.indptr)', len(s2a.indptr))
print('len(s2a.indices)', len(s2a.indices))
np.savetxt('s2a-indptr', s2a.indptr, '%d')
np.savetxt('s2a-indices', s2a.indices, '%d')
