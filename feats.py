# spark-submit --driver-memory 32g --executor-memory 32g --master local[*] --conf spark.driver.maxResultSize=32g feats.py tok2idx embeddings.npy RS RC

import pickle
import sys
from nltk.tokenize import word_tokenize
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window
import scipy.sparse as sps

indicator_t = np.int8
size_t = np.int32

def getter(field):
    return lambda row: row[field]

def indexer(field, tok2idx):
    def _indexer(row):
        return [tok2idx.get(tok, 0) for tok in word_tokenize(row[field])]
    return _indexer

def embed(rdd, indexer, embeddings):
    indexed_rdd = rdd.map(indexer).persist()
    indices = np.array(indexed_rdd.flatMap(lambda x: x).collect(), dtype=size_t)
    len_list = indexed_rdd.map(len).collect()
    indptr = np.cumsum(np.array([0] + len_list, dtype=size_t))
    data = np.ones(len(indices), dtype=indicator_t)
    row = len(indptr) - 1
    col = len(embeddings)
    csr = sps.csr_matrix((data, indices, indptr), dtype=indicator_t, shape=[row, col])
    len_array = np.expand_dims(np.array(len_list, dtype=size_t), 1)
    return csr.dot(embeddings) / len_array

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

sf = None
cf = None
for f in sys.argv[3:]:
    df = ss.read.format('json').load(f)
    if f.startswith('RS'):
        sf = df if sf is None else sf.union(df)
    elif f.startswith('RC'):
        cf = df if cf is None else cf.union(df)
    else:
        raise RuntimeError()

n_submissions = sf.count()
n_comments = cf.count()
cf = cf.withColumn('link_id', F.regexp_replace(cf.link_id, 't3_', '').cast(IntegerType()))

s_data = np.ones(n_submissions, dtype=indicator_t)
s_idx = np.arange(n_submissions, dtype=size_t)
s_sid = np.array(sf.rdd.map(getter('id')).collect(), dtype=size_t)
c_data = np.ones(n_comments, dtype=indicator_t)
c_sid = np.array(cf.rdd.map(getter('link_id')).collect(), dtype=size_t)
c_idx = np.arange(n_comments, dtype=size_t)
max_sid = max(np.max(s_sid), np.max(c_sid)) + 1
s_shape = [n_submissions, max_sid]
c_shape = [max_sid, n_comments]
s_idx2sid = sps.coo_matrix((s_data, (s_idx, s_sid)), dtype=indicator_t, shape=s_shape)
c_sid2idx = sps.coo_matrix((c_data, (c_sid, c_idx)), dtype=indicator_t, shape=c_shape)
s_idx2c_idx = s_idx2sid @ c_sid2idx

tok2idx = pickle.load(open(sys.argv[1], 'rb'))
embeddings = np.load(sys.argv[2])

sx = embed(sf.rdd, indexer('title', tok2idx), embeddings)
cx = embed(cf.rdd, indexer('body', tok2idx), embeddings)

minimum = np.ones([n_submissions, 1], dtype=size_t)
n_neighbors = s_idx2c_idx.sum(axis=1, dtype=size_t)
divisor = np.maximum(minimum, n_neighbors)
px = np.hstack([sx, s_idx2c_idx @ cx / divisor])
