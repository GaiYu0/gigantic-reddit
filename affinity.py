import argparse
import multiprocessing as mp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='src.npy')
parser.add_argument('--dst', type=str, default='dst.npy')
parser.add_argument('--y', type=str, default='y.npy')
parser.add_argument('--n-classes', type=int)
args = parser.parse_args()

with mp.Pool(3) as pool:
    src, dst, y = pool.map(np.load, [args.src, args.dst, args.y])

y_src = y[src]
y_dst = y[dst]
n = [np.sum(y == i) for i in range(args.n_classes)]
a = [[np.sum((y_src == i) & (y_dst == j)) / (n[i] * n[j])
      for j in range(args.n_classes)] for i in range(args.n_classes)]
np.set_printoptions(precision=3)
print(np.array(a))
