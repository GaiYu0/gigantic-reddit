import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='src.npy')
parser.add_argument('--dst', type=str, default='dst.npy')
parser.add_argument('--y', type=str, default='y.npy')
parser.add_argument('--n-classes', type=int)

src = np.load(args.src)
dst = np.load(args.dst)
y = np.load(args.y)

y_src = y[src]
y_dst = y[dst]
n = [np.sum(y == i) for i in range(args.n_classes)]
a = [[np.sum((y_src == i) & (y_dst == j)) / (n[i] * n[j])
      for j in range(args.n_classes)] for i in range(args.n_classes)]
print(a)
