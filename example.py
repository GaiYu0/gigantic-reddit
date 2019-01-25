import dgl
import mxnet as mx
import numpy as np

n_nodes = 8741
src = np.load('src.npy')
dst = np.load('dst.npy')
x = np.load('x.npy')
y = np.load('y.npy')
g = dgl.DGLGraph()
g.add_nodes(n_nodes)
g.add_edges(src, dst)
print(g.number_of_nodes())
print(g.number_of_edges())

print('mean:', np.mean(x))
print('std:', np.std(x))

K = 5
unique_y, counts = np.unique(y, return_counts=True)
index = np.argsort(counts)[-K:]
# Unmasked nodes are nodes in the largest K communities.
mask = np.isin(y, unique_y[index])
print(np.unique(y[mask], return_counts=True))

g.ndata['x'] = mx.nd.array(x)
g.ndata['y'] = mx.nd.array(y)
