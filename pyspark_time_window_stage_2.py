from collections import defaultdict
from functools import partial
import multiprocessing as mp
import pickle

from gluonnlp.data import SpacyTokenizer
import mxnet.ndarray as nd
import numpy as np
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType

from utils import fst, snd, loadtxt

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
post_df = ss.read.orc('post-df')

u = loadtxt(sc, 'u', int, np.int64)
v = loadtxt(sc, 'v', int, np.int64)
w = loadtxt(sc, 'w', int, np.int64)

uv = np.hstack([u, v])
unique, inverse = np.unique(uv, return_inverse=True)
nid = np.arange(len(unique))
src = nid[inverse[:len(u)]]
dst = nid[inverse[:len(v)]]
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
