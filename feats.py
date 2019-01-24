# spark-submit --driver-memory 32g --executor-memory 32g --master local[*] --conf spark.driver.maxResultSize=32g feats.py tok2idx embeddings.npy RS RC

from collections import defaultdict
import os
import operator
import pickle
import sys
import gluonnlp
from nltk.tokenize import word_tokenize
import mxnet as mx
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from utils import base36_decode, Timer

def getter(field):
    return lambda row: row[field]

def graph(sf, cf):
    sf = sf[['id', 'subreddit_id', 'title']].withColumnRenamed('id', 'sid')  # TODO integer
    cf = cf[['aid', 'body', 'link_id']].withColumn('sid', F.regexp_replace('link_id', 't3_', ''))

    sc = sf.join(cf, 'sid', 'inner').persist()

    ss = sc.alias('u').join(sc.alias('v'), 'aid', 'inner').persist()

    u = ss[['u.sid']].rdd.map(getter('sid')).map(base36_decode).collect()
    v = ss[['v.sid']].rdd.map(getter('sid')).map(base36_decode).collect()

    unique_u, inverse_u = np.unique(u, return_inverse=True)
    unique_v, inverse_v = np.unique(v, return_inverse=True)
    aau_u = np.argsort(np.argsort(unique_u))
    aau_v = np.argsort(np.argsort(unique_v))
    nid = np.arange(len(unique_u))
    src = nid[aau_u][inverse_u]
    dst = nid[aau_v][inverse_v]
    assert len(src) == len(dst)

    return len(nid), len(src), src, dst, sc, ss

nlp = 'gluonnlp' 
tokenizer = {
    'nltk'     : word_tokenize,
    'gluonnlp' : gluonnlp.data.SpacyTokenizer(),
}[nlp]

def embed_submissions(sc, ss, tok2idx, embeddings):
    def mapper(row):
        d = defaultdict(lambda: 0)
        for tok in tokenizer(row['title']):
            idx = tok2idx.get(tok, 0)
            d[idx] += 1
        indices = list(sorted(d.keys()))
        data = tuple(d[k] for k in indices)
        return indices, data

    ss = ss[['u.sid', 'u.title', 'u.subreddit_id']].distinct().sort('sid')
    rdd = ss.rdd.map(mapper).persist()
    data = rdd.flatMap(lambda x: x[1]).collect()
    indices = rdd.flatMap(lambda x: x[0]).collect()
    n = rdd.map(lambda x: len(x[0])).collect()
    indptr = np.hstack([np.zeros(1), np.cumsum(np.array(n))])
    shape = [len(indptr) - 1, len(embeddings)]
    csr_matrix = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
    n = mx.nd.maximum(1, mx.nd.expand_dims(mx.nd.array(n), axis=1))
    with Timer("sx = mx.nd.sparse.dot(csr_matrix, embeddings) / n"):
        sx = mx.nd.sparse.dot(csr_matrix, embeddings) / n
    return sx, ss

def embed_comments(sc, ss, tok2idx, embeddings):
    def flat_mapper(row):
        d = defaultdict(lambda: 0)
        for tok in tokenizer(row['body']):
            idx = tok2idx.get(tok, 0)
            d[idx] += 1
        sid = row['sid']
        if d:
            n = sum(d.values())
            return tuple(((sid, k), v / n) for k, v in d.items())
        else:
            return (((sid, 0), 0),)

    def mapper(x):
        (sid, k), v = x
        return (sid, (k, v))

    sc = sc.join(ss[['u.sid']], 'sid', 'inner')
    n = mx.nd.array(sc.groupBy('sid').count()[['count']].sort('sid').collect())
    rdd = sc.rdd.flatMap(flat_mapper)
    rdd = rdd.reduceByKey(operator.add)
    rdd = rdd.map(mapper)
    rdd = rdd.groupByKey()
    rdd = rdd.sortByKey()
    rdd = rdd.map(lambda x: tuple(zip(*x[1]))).persist()
    data = rdd.flatMap(lambda x: x[1]).collect()
    indices = rdd.flatMap(lambda x: x[0]).collect()
    indptr = np.hstack([np.zeros(1), np.cumsum(np.array(rdd.map(lambda x: len(x[0])).collect()))])
    shape = [len(indptr) - 1, len(embeddings)]
    csr_matrix = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
    with Timer("cx = mx.nd.sparse.dot(csr_matrix, embeddings) / n"):
        cx = mx.nd.sparse.dot(csr_matrix, embeddings) / n
    return cx, sc

def data(sc, ss, tok2idx, embeddings):
#   print(ss.rdd.map(getter('subreddit_id')).map(lambda x: x is None).reduce(operator.or_))
    sx, ss = embed_submissions(sc, ss, tok2idx, embeddings)
    cx, sc = embed_comments(sc, ss, tok2idx, embeddings)
    x = mx.nd.concat(sx, cx, dim=1).asnumpy()
    y = np.array(ss.withColumn('u.subreddit_id', F.regexp_replace('u.subreddit_id', 't5_', '')).rdd.map(getter('u.subreddit_id')).map(base36_decode).collect())
    return x, y

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

sf = None
cf = None
for f in sys.argv[4:]:
    df = ss.read.format('json').load(f)
    if os.path.basename(f).startswith('RS'):
        sf = df if sf is None else sf.union(df)
    elif os.path.basename(f).startswith('RC'):
        cf = df if cf is None else cf.union(df)
    else:
        raise RuntimeError()
author = ss.read.format('json').load(sys.argv[3])[['name', 'id']] \
    .withColumnRenamed('name', 'author') \
    .withColumn('aid', F.regexp_replace('id', 't2_', ''))  # TODO integer
cf = cf.join(author, 'author', 'inner')

n_submissions = sf.count()
n_comments = cf.count()
print('# submissions:', n_submissions)
print('# comments:', n_comments)

n_nodes, n_edges, src, dst, sc, ss = graph(sf, cf)
print('# nodes:', n_nodes)
print('# edges:', n_edges)
pickle.dump(n_nodes, open('n_nodes', 'wb'))
np.save('src', src)
np.save('dst', dst)

tok2idx = pickle.load(open(sys.argv[1], 'rb'))
embeddings = mx.nd.array(np.load(sys.argv[2]))

x, y = data(sc, ss, tok2idx, embeddings)
unique_y, inverse_y = np.unique(y, return_inverse=True)
y = np.arange(len(unique_y))[inverse_y]
print('Features:', x.shape)
print('Labels:', y.shape)
np.save('x', x)
np.save('y', y)
