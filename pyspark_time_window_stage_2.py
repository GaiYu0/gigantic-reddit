from collections import defaultdict
from functools import partial
import multiprocessing as mp
import pickle

from gluonnlp.data import SpacyTokenizer
import mxnet.ndarray as nd
import numpy as np
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType

from utils import fst, snd

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
post_df = ss.read.orc('post-df')

with mp.Pool(3) as pool:
    u, v, w = pool.map(partial(np.loadtxt, dtype=np.int64), ['u', 'v', 'w'])
pid = np.array(post_df.select('pid').rdd.map(lambda x: x).collect())
isin = np.isin(u, pid) & np.isin(v, pid)
with mp.Pool(3) as pool:
    u, v, w = pool.starmap(np.ndarray.__getitem__, [[u, isin], [v, isin], [w, isin]])

with mp.Pool(2) as pool:
    [unique_u, inverse_u], [unique_v, inverse_v] = pool.map(partial(np.unique, return_inverse=True), [u, v])
unique, inverse = np.unique(np.hstack([unique_u, unique_v]), return_inverse=True)
nid = np.arange(len(unique))
def f(x):
    return nid[inverse][:len(unique_u)][inverse_u] if x else nid[inverse][len(unique_u):][inverse_v]
with mp.Pool(2) as pool:
    src, dst = pool.map(f, [True, False])
post_df = post_df.join(sc.parallelize(zip(map(int, unique), map(int, nid))).toDF(['pid', 'nid']), 'pid')

tok2idx = pickle.load(open('tok2idx', 'rb'))
tokenizer = SpacyTokenizer()
def tokenize(x):
    d = defaultdict(lambda: 0)
    for tok in tokenizer(x.title):
        idx = tok2idx.get(tok, 0)  # TODO default value
        d[idx] += 1
    if d:
        n = sum(d.values())
        return [[x.nid, len(d)], [[k, d[k] / n] for k in sorted(d)]]
    else:
        return [[x.nid, 0], tuple()]

embeddings = nd.array(np.load('embeddings.npy'))
x_rdd = post_df.select('nid', 'title').rdd.map(tokenize)
indptr  = nd.array(np.cumsum([0] + x_rdd.map(fst).map(snd).collect()))
indices = nd.array(x_rdd.map(snd).flatMap(lambda x: x).map(fst).collect())
data = nd.array(x_rdd.map(snd).flatMap(lambda x: x).map(snd).collect())
shape = [len(indptr) - 1, len(embeddings)]
matrix = nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
x = nd.sparse.dot(matrix, embeddings).asnumpy()[np.argsort(x_rdd.map(fst).map(fst).collect())]

y_rdd = post_df.select('nid', 'srid').rdd
y = np.array(y_rdd.map(lambda d: d['srid']).collect())[np.argsort(y_rdd.map(lambda d: d['nid']).collect())]
unique_y, inverse_y = np.unique(y, return_inverse=True)
y = np.arange(len(unique_y))[inverse_y]

np.save('x', x)
np.save('y', y)
np.save('src', src)
np.save('dst', dst)