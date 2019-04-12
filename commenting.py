import pickle
import numpy as np

[[a, b, c, d],
 [e, f, g, h], _] = pickle.load(open('layers', 'rb'))

u = np.load('u.npy')
v = np.load('v.npy')
print(u.shape, v.shape)

cau = u[g : h]
cav = v[g : h]

sau = cau - c + (c - b)
sav = u[cav] - b

print(d - b)
np.save('sau', sau)
np.save('sav', sav)
