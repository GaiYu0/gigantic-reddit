import numpy as np
import scipy.sparse as sps

sau = np.load('sau.npy')
sav = np.load('sav.npy')
m = len(sau)
n = max(np.max(sau), np.max(sav)) + 1

print(sau.min(), sau.max())
print(sav.min(), sav.max())

a2s = sps.coo_matrix((np.ones(m), (sau, sav)), shape=(n, n)).tocsr()
a2s_indptr = a2s.indptr[sau.min():]
a2s_indices = a2s.indices
print('len(a2s.indptr)', len(a2s_indptr))
print('len(a2s.indices)', len(a2s_indices))
np.savetxt('a2s-indptr', a2s_indptr, '%d')
np.savetxt('a2s-indices', a2s_indices, '%d')

s2a = sps.coo_matrix((np.ones(m), (sav, sau)), shape=(n, n)).tocsr()
s2a_indptr = s2a.indptr[:sau.min() + 1]
s2a_indices = s2a.indices - sau.min()
assert np.all(s2a_indptr[:-1] <= s2a_indptr[1:])
print('len(s2a.indptr)', len(s2a_indptr))
print('len(s2a.indices)', len(s2a_indices))
np.savetxt('s2a-indptr', s2a_indptr, '%d')
np.savetxt('s2a-indices', s2a.indices - sau.min(), '%d')
