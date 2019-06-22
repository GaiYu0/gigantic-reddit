import numpy as np
import scipy.sparse as sps
from pyspark.sql.session import SparkSession

def loadtxt(sc, fname, convert, dtype):
    return np.array(sc.textFile(fname, use_unicode=False).map(convert).collect(), dtype=dtype)

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext

p2u_indptr = loadtxt(sc, 'p2u-indptr', int, np.int64)
p2u_indices = loadtxt(sc, 'p2u-indices', int, np.int64)
u2p_indptr = loadtxt(sc, 'u2p-indptr', int, np.int64)
u2p_indices = loadtxt(sc, 'u2p-indices', int, np.int64)
m = len(p2u_indptr) - 1
n = len(u2p_indptr) - 1
p2u = sps.csr_matrix((np.ones_like(p2u_indices), p2u_indices, p2u_indptr), shape=(m, n))
u2p = sps.csr_matrix((np.ones_like(u2p_indices), u2p_indices, u2p_indptr), shape=(n, m))
p2p = p2u @ u2p

idx = p2p.indptr[1 : -1]
rdd = sc.parallelize(zip(np.split(p2p.indices, idx), np.split(p2p.data, idx)))
def top_k(x, k=5):
    indices, data = x
    return indices[np.argsort(data)[-k:]]
rdd = rdd.map(top_k)
indptr = np.array(np.cumsum([0] + rdd.map(len).collect()))
indices = np.array(rdd.flatMap(lambda x: x).collect())
a = sps.csr_matrix((np.ones_like(indices), indices, indptr), shape=(m, m))
b = a.maximum(a.transpose()).tocoo()

np.save('src', b.col)
np.save('dst', b.row)
