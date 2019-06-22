import numpy as np
from pyspark.sql.session import SparkSession
import scipy.sparse as sps

import utils

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext

src = utils.loadtxt(sc, 'src', int, np.int64)
dst = utils.loadtxt(sc, 'dst', int, np.int64)
n = np.max(src) + 1
a = sps.coo_matrix((np.ones_like(src), (src, dst)), [n, n])
b = a.maximum(a.transpose()).tocoo()

np.save('src', b.row)
np.save('dst', b.col)
