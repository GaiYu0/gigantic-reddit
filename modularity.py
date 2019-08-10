import argparse
import numpy as np
import scipy.sparse as sps

def modularity(src, dst, y, n_classes=None, p=False):
    n = len(y)
    m = len(src)
    n_classes = len(np.unique(y)) if n_classes is None else n_classes

    adj = sps.coo_matrix((np.ones(len(src)), (src, dst)), [n, n])
    d = np.array(adj.sum(1))
    mod = adj - sps.coo_matrix((d[src] * d[dst] / (2 * m - 1), (src, dst)), [n, n])
    s = np.zeros([n, n_classes])
    s[np.arange(n), y] = 1
    return np.sum(s * (s @ mod))

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

    print(modularity(src, dst, y, args.n_classes, args.p))
