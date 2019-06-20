import argparse
import dask
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='src.npy')
parser.add_argument('--dst', type=str, default='dst.npy')
parser.add_argument('--y', type=str, default='y.npy')
parser.add_argument('--n-classes', type=int)
args = parser.parse_args()

src = np.load(args.src)
dst = np.load(args.dst)
y = np.load(args.y)

y_src = y[src]
y_dst = y[dst]
ns = dask.compute(*[dask.delayed(lambda i: np.sum(y == i))(i) for i in range(args.n_classes)])
print(ns)
a = dask.compute([[dask.delayed(lambda i, j, m, n: np.sum((y_src == i) & (y_dst == j)) / (m * n))(i, j, m, n) for j, n in enumerate(ns)] for i, m in enumerate(ns)])
# np.set_printoptions(precision=3)
print(np.array(a))
