import argparse
import multiprocessing as mp
import numpy as np
import scipy.sparse as sps

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='src.npy')
parser.add_argument('--dst', type=str, default='dst.npy')
parser.add_argument('--y', type=str, default='y.npy')
args = parser.parse_args()

with mp.Pool(3) as pool:
    src, dst, y = pool.map(np.load, [args.src, args.dst, args.y])

n = len(y)
dat = np.ones_like(src)
a = sps.coo_matrix((dat, (src, dst)), shape=(n, n))
assert not np.any((a != a.transpose()).data)
