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
from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession
import utils

def graph(sf, cf, af):
    sf = sf.select('id', 'subreddit_id', 'title') \
           .withColumnRenamed('id', 'sid') \
           .withColumn('sid', utils.udf_int36('sid')) \
           .withColumnRenamed('subreddit_id', 'srid')  # sid, srid, title
    af = af.select('name', 'id') \
           .withColumnRenamed('name', 'author') \
           .withColumn('aid', regexp_replace('id', 't2_', '')) \
           .withColumn('aid', utils.udf_int36('aid'))  # aid, author
    cf = cf.join(af, 'author', 'inner') \
           .select('aid', 'body', 'link_id') \
           .withColumn('sid', regexp_replace('link_id', 't3_', '')) \
           .withColumn('sid', utils.udf_int36('sid'))  # aid, body, sid

    sc = sf.join(cf, 'sid', 'inner').persist()

    u = sc.alias('u')
    v = sc.alias('v')
    ss = u.join(v, 'aid', 'inner').persist()

    u = ss.select('u.sid').rdd.map(utils.getter('sid')).collect()
    v = ss.select('v.sid').rdd.map(utils.getter('sid')).collect()

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
        if d:
            n = sum(d.values())
            indices = tuple(sorted(d.keys()))
            data = tuple(d[k] / n for k in indices)
            return indices, data
        else:
            return (0,), (0,)  # TODO randomize index

    ss = ss.select('u.sid', 'u.title', 'u.srid').distinct().sort('sid')
    rdd = ss.rdd.map(mapper).persist()
    data = rdd.flatMap(utils.snd).collect()
    indices = rdd.flatMap(utils.fst).collect()
    rdd = rdd.map(utils.fst)
    rdd = rdd.map(len)
    indptr = np.hstack([np.zeros(1), np.cumsum(np.array(rdd.collect()))])
    shape = [len(indptr) - 1, len(embeddings)]
    csr_matrix = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
    with utils.Timer("sx = mx.nd.sparse.dot(csr_matrix, embeddings)"):
        sx = mx.nd.sparse.dot(csr_matrix, embeddings)
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
    rdd = sc.rdd.flatMap(flat_mapper) \
                .reduceByKey(operator.add) \
                .map(mapper) \
                .groupByKey() \
                .sortByKey() \
                .map(utils.snd) \
                .map(utils.starzip) \
                .map(tuple).persist()
    data = rdd.flatMap(utils.snd).collect()
    indices = rdd.flatMap(utils.fst).collect()
    rdd = rdd.map(utils.fst).map(len)
    indptr = np.hstack([np.zeros(1), np.cumsum(np.array(rdd.collect()))])
    shape = [len(indptr) - 1, len(embeddings)]
    csr_matrix = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
    df = sc.groupBy('sid').count().sort('sid').select('count')
    n = mx.nd.array(df.collect())
    with utils.Timer("cx = mx.nd.sparse.dot(csr_matrix, embeddings) / n"):
        cx = mx.nd.sparse.dot(csr_matrix, embeddings) / n
    return cx, sc

def labels(ss):
    rdd = ss.rdd.map(utils.getter('srid'))
    rdd = rdd.map(lambda x: x.replace('t5_', ''))
    rdd = rdd.map(utils.int36)
    y = np.array(rdd.collect())
    return y

def data(sc, ss, tok2idx, embeddings):
#   print(ss.rdd.map(utils.getter('srid')).map(lambda x: x is None).reduce(operator.or_))
    sx, ss = embed_submissions(sc, ss, tok2idx, embeddings)
    cx, sc = embed_comments(sc, ss, tok2idx, embeddings)
    x = mx.nd.concat(sx, cx, dim=1).asnumpy()
    y = labels(ss)
    return x, y

session = SparkSession.builder.getOrCreate()
context = session.sparkContext
logger = context._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

sf = None
cf = None
for f in sys.argv[4:]:
    df = session.read.format('json').load(f)
    if os.path.basename(f).startswith('RS'):
        sf = df if sf is None else sf.union(df)
    elif os.path.basename(f).startswith('RC'):
        cf = df if cf is None else cf.union(df)
    else:
        raise RuntimeError()
af = session.read.format('json').load(sys.argv[3])

n_submissions = sf.count()
n_comments = cf.count()
n_authors = af.count()
print('# submissions:', n_submissions)
print('# comments:', n_comments)

n_nodes, n_edges, src, dst, sc, ss = graph(sf, cf, af)
print('# nodes:', n_nodes)
print('# edges:', n_edges)
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
