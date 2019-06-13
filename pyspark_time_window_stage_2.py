from functools import partial
import multiprocessing as mp
import pickle

import mxnet.ndarray as nd
import numpy as np
from pyspark.sql.session import sparksession

ss = SparkSession.builder.getOrCreate()
post_df = ss.read.orc('post-df')

with mp.Pool(3) as pool:
    u, v, w = pool.map(partial(np.loadtxt, dtype=np.int64), ['u', 'v', 'w'])
pid = np.array(post_df.select('pid').rdd.map(lambda x: x).collect())
isin = u.isin(pid) & v.isin(pid)
with mp.Pool(3) as pool:
    u, v, w = pool.map(np.ndarray.__getitem__, [u, v, w])

with mp.Pool(2) as pool:
    [unique_u, inverse_u], [unique_v, inverse_v] = pool.map(partial(np.unique, return_inverse=True), [u, v])
unique, inverse = np.unique(np.hstack([unique_u, unique_v]), return_inverse=True)
nid = np.arange(len(unique))
with mp.Pool(2) as pool:
    src, dst = pool.map(partial(lambda x: nid[inverse][:len(unique_u)][inverse_u] if x else nid[inverse][len(unique_u):][inverse_v]), [True, False])
pid_nid_df = ss.createDataFrame(zip(unique, nid))
               .withColumnRenamed('_1', 'pid')
               .withColumnRenamed('_2', 'nid')

tok2idx = pickle.load(open('tok2idx', 'rb'))
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

post_rdd = post_df.join(pid_nid_df, ['pid'])
                  .select('nid', 'title').rdd
post_rdd = post_rdd.map(tokenize)
indptr  = nd.array(np.cumsum([0] + post_rdd.map(fst).map(snd).collect()))
indices = nd.array(post_rdd.map(snd).flatMap(lambda x: x).map(fst).collect())
data = nd.array(post_rdd.map(snd).flatMap(lambda x: x).map(snd).collect())
shape = [len(indptr) - 1, len(embeddings)]
matrix = nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
embeddings = nd.array(np.load('embeddings'))
z = matrix @ embeddings
x = z[np.argsort(post_rdd.map(fst).map(fst))]

np.save('x', x)
np.save('src', src)
np.save('dst', dst)
