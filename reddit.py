import operator
import pickle
import sys
from nltk.tokenize import word_tokenize
import numpy as np
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import Row
from pyspark.sql.session import SparkSession

ss = SparkSession.builder.master('local').appName('reddit').config('spark.ui.showConsoleProgress', True).getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

'''
df = ss.read.format('json').load(sys.argv[2])
rdd = df[['id', 'parent_id']].rdd
rdd = rdd.map(lambda row: (row['id'], row['parent_id']))
rdd = rdd.groupWith(rdd)
rdd = rdd.flatMap(lambda kvw: [((v, w), 1) for v in kvw[1][0] for w in kvw[1][1]])
rdd.reduceByKey(operator.add)
print(rdd.count())
'''

def fn(row):
    tok = word_tokenize(row['body'])
    return [Row(parent_id=row['parent_id'], idx=tok2idx[t]) for t in tok]

df = ss.read.format('json').load(sys.argv[2])
rdd = df[['parent_id', 'body']].rdd
tok2idx = pickle.load(open(sys.argv[3], 'rb'))
rdd = rdd.flatMap(fn)
df = rdd.toDF()

embeddings = np.load(sys.argv[4])
embeddings = sc.parallelize([(i, embedding) for i, embedding in enumerate(embeddings)]).toDF('idx', 'embedding')
df.join(embeddings, on='idx', how='inner').groupBy('parent_id').avg('embedding')

# df = ss.read.format('json').load(sys.argv[2])
