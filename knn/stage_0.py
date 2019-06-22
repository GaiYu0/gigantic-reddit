from collections import defaultdict
import pickle

import dask
from gluonnlp.data import SpacyTokenizer
import mxnet.ndarray as nd
import numpy as np
from pyspark.sql.session import SparkSession

from utils import fst, snd, loadtxt

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext

cmnt_df = ss.read.orc('cmnt-df')
user_df = cmnt_df.select('uid').dropDuplicates()
post_df = ss.read.orc('post-df').join(cmnt_df.select('pid'), 'pid').dropDuplicates()

collect_column = lambda df, column: df.select(column).rdd.flatMap(lambda x: x).collect()
cmnt_df = cmnt_df.join(sc.parallelize(zip(collect_column(post_df, 'pid'), range(post_df.count()))).toDF(['pid', 'compact_pid']), 'pid') \
                 .join(sc.parallelize(zip(collect_column(user_df, 'uid'), range(user_df.count()))).toDF(['uid', 'compact_uid']), 'uid')

p2u = cmnt_df.rdd.map(lambda row: [row.compact_pid, row.compact_uid]).groupByKey().sortByKey()
p2u_indptr = np.cumsum([0] + p2u.map(lambda kv: len(kv[1])).collect())
p2u_indices = np.array(p2u.flatMap(lambda kv: kv[1]).collect())

u2p = cmnt_df.rdd.map(lambda row: [row.compact_uid, row.compact_pid]).groupByKey().sortByKey()
u2p_indptr = np.cumsum([0] + u2p.map(lambda kv: len(kv[1])).collect())
u2p_indices = np.array(u2p.flatMap(lambda kv: sorted(kv[1])).collect())

print('# posts:', len(p2u_indptr) - 1)
print('# users:', len(u2p_indptr) - 1)
print('# cmnts:', len(p2u_indices))
dask.compute(dask.delayed(np.savetxt)('p2u-indptr', p2u_indptr, fmt='%d'),
             dask.delayed(np.savetxt)('p2u-indices', p2u_indices, fmt='%d'),
             dask.delayed(np.savetxt)('u2p-indptr', u2p_indptr, fmt='%d'),
             dask.delayed(np.savetxt)('u2p-indices', u2p_indices, fmt='%d'))

post_df = post_df.join(cmnt_df.select('pid', 'compact_pid').dropDuplicates(), 'pid')

tok2idx = pickle.load(open('tok2idx', 'rb'))
tokenizer = SpacyTokenizer()
def tokenize(x):
    d = defaultdict(lambda: 0)
    for tok in tokenizer(x.title):
        idx = tok2idx.get(tok, 0)  # TODO default value
        d[idx] += 1
    if d:
        n = sum(d.values())
        return [[x.compact_pid, len(d)], [[k, d[k] / n] for k in sorted(d)]]
    else:
        return [[x.compact_pid, 0], tuple()]

embeddings = nd.array(np.load('embeddings.npy'))
x_rdd = post_df.rdd.map(tokenize)
indptr  = nd.array(np.cumsum([0] + x_rdd.map(fst).map(snd).collect()))
indices = nd.array(x_rdd.map(snd).flatMap(lambda x: x).map(fst).collect())
data = nd.array(x_rdd.map(snd).flatMap(lambda x: x).map(snd).collect())
shape = [len(indptr) - 1, len(embeddings)]
matrix = nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
x = nd.sparse.dot(matrix, embeddings).asnumpy()[np.argsort(x_rdd.map(fst).map(fst).collect())]

y = np.array(post_df.rdd.map(lambda row: [row.compact_pid, row.srid]).sortByKey().map(lambda kv: kv[1]).collect())
unique, inverse = np.unique(y, return_inverse=True)
y = np.arange(len(unique))[inverse]
np.save('y', y)
