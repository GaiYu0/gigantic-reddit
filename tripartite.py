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
    """
    Problems
    --------
        Duplicated author names.
    """
    af = af.select('name', 'id') \
           .withColumnRenamed('id', 'aid') \
           .withColumnRenamed('name', 'author') \
           .dropDuplicates(['author']) \
           .withColumn('aid', regexp_replace('aid', 't2_', '')) \
           .withColumn('aid', utils.udf_int36('aid'))
    cf = cf.select('author', 'body', 'link_id') \
           .withColumnRenamed('link_id', 'sid') \
           .withColumn('sid', regexp_replace('sid', 't3_', '')) \
           .withColumn('sid', utils.udf_int36('sid'))
    sf = sf.select('author', 'id', 'subreddit_id', 'title') \
           .withColumnRenamed('id', 'sid') \
           .withColumnRenamed('subreddit_id', 'srid') \
           .withColumn('sid', utils.udf_int36('sid')) \
           .withColumn('srid', regexp_replace('srid', 't2_', '')) \
           .withColumn('srid', utils.udf_int36('srid'))         

    caf = cf.select('author', 'sid') \
            .join(af, 'author') \
            .drop('author') \
            .withColumnRenamed('aid', 'caid')
    saf = sf.select('author', 'sid') \
            .join(af, 'author') \
            .drop('author') \
            .withColumnRenamed('aid', 'said')
    csf = caf.join(saf, 'sid').dropDuplicates(['sid'])
    caf = csf.select('sid', 'caid').withColumnRenamed('caid', 'aid')
    saf = csf.select('sid', 'said').withColumnRenamed('said', 'aid')
    cf = cf.drop('author').join(caf, 'sid')
    sf = sf.drop('author').join(saf, 'sid')

    n_comments = cf.count()
    n_submissions = sf.count()
    cnid = np.arange(n_comments)
    snid = np.arange(n_submissions) + n_comments

    arange2range = lambda x: range(x[0], x[-1] + 1)

    sid = utils.column2list(sf[['sid']])
    sid2snid = context.parallelize(zip(sid, arange2range(snid))).toDF() \
                      .withColumnRenamed('_1', 'sid') \
                      .withColumnRenamed('_2', 'snid')
    csu = np.array(utils.column2list(cf.join(sid2snid, 'sid')[['snid']]))
    csv = cnid
    print(csu.shape, csv.shape)

    caid = utils.column2rdd(cf[['aid']])
    said = utils.column2rdd(sf[['aid']])
    aid = caid.union(said).distinct().collect()
    anid = np.arange(len(aid)) + n_comments + n_submissions
    aid2anid = context.parallelize(zip(aid, arange2range(anid))).toDF() \
                      .withColumnRenamed('_1', 'aid') \
                      .withColumnRenamed('_2', 'anid')

    sau = np.array(utils.column2list(sf.join(aid2anid, 'aid', 'inner')[['anid']]))
    sav = snid
    cau = np.array(utils.column2list(cf.join(aid2anid, 'aid', 'inner')[['anid']]))
    cav = cnid

    node_offsets = [0, len(cnid),
                       len(cnid) + len(snid),
                       len(cnid) + len(snid) + len(anid)]
    edge_offsets = [0, len(csu),
                       len(csu) + len(sau),
                       len(csu) + len(sau) + len(cau)]
    u = np.hstack([csu, sau, cau])
    v = np.hstack([csv, sav, cav])
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
    with utils.Timer('%s: %d' % (utils.filename(), utils.lineno())):
        x = mx.nd.sparse.dot(csr_matrix, embeddings)
    return x.asnumpy()

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
with utils.Timer('%s: %d' % (utils.filename(), utils.lineno())):
    np.save('u', u)
with utils.Timer('%s: %d' % (utils.filename(), utils.lineno())):
    np.save('v', v)

tok2idx = pickle.load(open(sys.argv[1], 'rb'))
embeddings = mx.nd.array(np.load(sys.argv[2]))
cx = embed_nodes(utils.column2rdd(cf[['body']]), tok2idx, embeddings)
sx = embed_nodes(utils.column2rdd(sf[['title']]), tok2idx, embeddings)
with utils.Timer('%s: %d' % (utils.filename(), utils.lineno())):
    np.save('cx', cx)
with utils.Timer('%s: %d' % (utils.filename(), utils.lineno())):
    np.save('sx', sx)

sy = np.array(utils.column2rdd(sf[['srid']]).collect())
sy = np.array(utils.column2rdd(cf.join(sf, 'sid', 'inner')[['srid']]).collect())
with utils.Timer('%s: %d' % (utils.filename(), utils.lineno())):
    np.save('sy', sy)
