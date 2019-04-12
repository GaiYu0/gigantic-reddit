import pickle

import numpy as np
import scipy.sparse as sps

[[a, b, c, d],
 [e, f, g, h], _] = pickle.load(open('layers', 'rb'))
print(a, b, c, d)
print(e, f, g, h)

n_comments = b - a
n_submissions = c - b
n_authors = d - c
print(n_comments, n_submissions, n_authors)

u = np.load('u.npy')
v = np.load('v.npy')
print(u.shape, v.shape)

ca_u = u[g : h]
ca_v = v[g : h]
A = sps.coo_matrix((np.ones(len(ca_u)), (ca_u, ca_v)), shape=(d, d))
B = (A.transpose() @ A).tocoo()
cc_u = B.row
cc_v = B.col
ss_u = u[cc_u] - n_comments
ss_v = u[cc_v] - n_comments
print(ss_u.shape, ss_v.shape)

np.save('ss_u', ss_u)
np.save('ss_v', ss_v)

'''
sa_u = u[p : q]
sa_v = v[p : q]

A = sps.coo_matrix((np.ones(len(sa_u)), (sa_u, sa_v)), shape=(n, n))
B = (A.transpose() @ A).tocoo()
ss_u = B.row - n_comments
ss_v = B.col - n_comments
print(ss_u.min(), ss_u.max())
print(ss_v.min(), ss_v.max())
np.save('ss_u', ss_u)
np.save('ss_v', ss_v)
'''
