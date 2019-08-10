import argparse
import dask
import numpy as np

def affinity_matrix(src, dst, y, n_classes=None, p=False):
    y_src = y[src]
    y_dst = y[dst]
    n_classes = len(np.unique(y)) if n_classes is None else n_classes
    ns = dask.compute(*[dask.delayed(lambda i: np.sum(y == i))(i) for i in range(n_classes)])
    affinity = lambda i, j, m, n: np.sum((y_src == i) & (y_dst == j)) / (m * n if p else 1)
    return np.array(dask.compute([[dask.delayed(affinity)(i, j, m, n) \
                                   for j, n in enumerate(ns)] for i, m in enumerate(ns)]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='src.npy')
    parser.add_argument('--dst', type=str, default='dst.npy')
    parser.add_argument('--n-classes', type=int)
    parser.add_argument('-p', action='store_true')
    parser.add_argument('--y', type=str, default='y.npy')
    args = parser.parse_args()

    src = np.load(args.src)
    dst = np.load(args.dst)
    y = np.load(args.y)

    np.set_printoptions(precision=3)
    print(affinity_matrix(src, dst, y, args.n_classes, args.p))
