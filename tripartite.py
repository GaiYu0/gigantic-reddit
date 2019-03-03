# TODO connected components?
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

def construct_graph(context, cf, sf, af):
    af = af.select('id', 'name') \
           .withColumnRenamed('id', 'aid') \
           .withColumnRenamed('name', 'author') \
           .withColumn('aid', regexp_replace('aid', 't2_', '')) \
           .withColumn('aid', utils.udf_int36('aid'))  # aid, author
    cf = cf.join(af, 'author', 'inner') \
           .select('aid', 'body', 'link_id') \
           .withColumnRenamed('link_id', 'sid') \
           .withColumn('sid', regexp_replace('sid', 't3_', '')) \
           .withColumn('sid', utils.udf_int36('sid'))  # aid, body, sid
    sf = sf.join(af, 'author', 'inner') \
           .select('aid', 'id', 'subreddit_id', 'title') \
           .withColumnRenamed('id', 'sid') \
           .withColumnRenamed('subreddit_id', 'srid') \
           .withColumn('sid', utils.udf_int36('sid'))  # aid, sid, srid, title

    n_comments = cf.count()
    n_submissions = sf.count()
    cnid = np.arange(n_comments)
    snid = np.arange(n_submissions) + n_comments

    arange2range = lambda x: range(x[0], x[-1] + 1)

    sid = sf.select('sid').rdd.map(utils.getter('sid')).collect()
    sid2snid = context.parallelize(zip(sid, arange2range(snid))).toDF() \
                      .withColumnRenamed('_1', 'sid') \
                      .withColumnRenamed('_2', 'snid')
    cs_u = np.array(utils.column2list(cf.join(sid2snid, 'sid', 'inner')[['snid']]))
    cs_v = cnid

    said = np.array(utils.column2list(sf[['aid']]))
    caid = np.array(utils.column2list(sf[['aid']]))
    aid = np.unique(np.hstack([said, caid]))
    anid = np.arange(len(aid)) + n_comments + n_submissions
    aid2anid = context.parallelize(zip(arange2range(aid), arange2range(anid))).toDF() \
                      .withColumnRenamed('_1', 'aid') \
                      .withColumnRenamed('_2', 'anid')

    sa_u = np.array(utils.column2list(sf.join(aid2anid, 'aid', 'inner')[['anid']]))
    sa_v = snid
    ca_u = np.array(utils.column2list(cf.join(aid2anid, 'aid', 'inner')[['anid']]))
    ca_v = cnid

    node_offsets = [0, len(cnid),
                       len(cnid) + len(snid),
                       len(cnid) + len(snid) + len(anid)]
    edge_offsets = [0, len(cs_u),
                       len(cs_u) + len(sa_u),
                       len(cs_u) + len(sa_u) + len(ca_u)]
    u = np.hstack([cs_u, sa_u, ca_u])
    v = np.hstack([cs_v, sa_v, ca_v])
    type2lid = {'comment' : 0, 'submission' : 1, 'author' : 2}
    return node_offsets, edge_offsets, u, v, type2lid, cf, sf

nlp = 'gluonnlp' 
tokenizer = {
    'nltk'     : word_tokenize,
    'gluonnlp' : gluonnlp.data.SpacyTokenizer(),
}[nlp]

def embed_nodes(rdd, tok2idx, embeddings):
    def mapper(row):
        d = defaultdict(lambda: 0)
        for tok in tokenizer(row):
            idx = tok2idx.get(tok, 0)  # TODO default value
            d[idx] += 1
        if d:
            n = sum(d.values())
            return (len(d), tuple((k, d[k] / n) for k in sorted(d)))
        else:
            return (0, tuple())

    rdd = rdd.map(mapper).persist()
    indptr = mx.nd.array(np.cumsum(np.array([0] + rdd.map(utils.fst).collect())))  # TODO insertion
    rdd = rdd.map(utils.snd).flatMap(lambda x: x).persist()
    indices = mx.nd.array(rdd.map(utils.fst).collect())
    data = mx.nd.array(rdd.map(utils.snd).collect())
    shape = [len(indptr) - 1, len(embeddings)]
    csr_matrix = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
    x = mx.nd.sparse.dot(csr_matrix, embeddings)
    return x

session = SparkSession.builder.getOrCreate()
context = session.sparkContext
logger = context._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

cf = None
sf = None
for f in sys.argv[4:]:
    df = session.read.format('json').load(f)
    if os.path.basename(f).startswith('RC'):
        cf = df if cf is None else cf.union(df)
    elif os.path.basename(f).startswith('RS'):
        sf = df if sf is None else sf.union(df)
    else:
        raise RuntimeError()
af = session.read.format('json').load(sys.argv[3])

node_offsets, edge_offsets, u, v, type2lid, cf, sf = construct_graph(context, cf, sf, af)
pickle.dump([node_offsets, edge_offsets, type2lid], open('layers', 'wb'))
np.save('u', u)
np.save('v', v)

tok2idx = pickle.load(open(sys.argv[1], 'rb'))
embeddings = mx.nd.array(np.load(sys.argv[2]))
cx = embed_nodes(utils.column2rdd(cf[['body']]), tok2idx, embeddings)
sx = embed_nodes(utils.column2rdd(sf[['title']]), tok2idx, embeddings)
np.save('cx', cx)
np.save('sx', sx)

rdd = sf.rdd.map(utils.getter('srid')) \
            .map(lambda x: x.replace('t5_', '')) \
            .map(utils.int36)
y = np.array(rdd.collect())
# TODO cy
np.save('sy', y)
