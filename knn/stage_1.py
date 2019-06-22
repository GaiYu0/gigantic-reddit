import numpy as np
import scipy.sparse as sps
from pyspark.sql.session import SparkSession

import utils

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext

p2u_indptr = utils.loadtxt(sc, 'p2u-indptr', int, np.int64)
p2u_indices = utils.loadtxt(sc, 'p2u-indices', int, np.int64)
u2p_indptr = utils.loadtxt(sc, 'u2p-indptr', int, np.int64)
u2p_indices = utils.loadtxt(sc, 'u2p-indices', int, np.int64)
m = len(p2u_indptr) - 1
n = len(u2p_indptr) - 1
p2u = sps.csr_matrix((np.ones_like(p2u_indices), p2u_indices, p2u_indptr), shape=(m, n))
u2p = sps.csr_matrix((np.ones_like(u2p_indices), u2p_indices, u2p_indptr), shape=(n, m))
p2p = p2u @ u2p

idx = p2p.indptr[1 : -1]
rdd = ss.parallelize(zip(np.split(p2p.indices, idx), np.split(p2p.data, idx)))
k = 5
rdd = rdd.map(lambda indices, data: indices[np.argsort(data)[-k:]])
indptr = np.array(rdd.map(len).collect())
indices = np.array(rdd.flatMap(lambda x: x).collect())
a = sps.csr_matrix((np.ones_like(indices), indices, indptr), shape=(m, m))
b = a.maximum(a.transpose())
print('end')
