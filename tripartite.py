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
    af = af.select('name', 'id') \
           .withColumnRenamed('name', 'author') \
           .withColumn('aid', regexp_replace('id', 't2_', '')) \
           .withColumn('aid', utils.udf_int36('aid'))  # aid, author
    sf = sf.join(af, 'author', 'inner') \
           .select('aid', 'id', 'subreddit_id', 'title') \
           .withColumnRenamed('id', 'sid') \
           .withColumnRenamed('subreddit_id', 'srid') \
           .withColumn('sid', utils.udf_int36('sid'))  # aid, sid, srid, title
    cf = cf.join(af, 'author', 'inner') \
           .select('aid', 'body', 'link_id') \
           .withColumnRenamed('link_id', 'sid') \
           .withColumn('sid', regexp_replace('sid', 't3_', '')) \
           .withColumn('sid', utils.udf_int36('sid'))  # aid, body, sid

    sid = sf.select('sid').rdd.map(utils.getter('sid')).collect()
    snid = range(len(sid))
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
