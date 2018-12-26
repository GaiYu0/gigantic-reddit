import operator
import pickle
import sys
from nltk.tokenize import word_tokenize
import numpy as np
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import Row
from pyspark.sql.session import SparkSession

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

'''
df = ss.read.format('json').load(sys.argv[1])
rdd = df[['id', 'parent_id']].rdd
rdd = rdd.map(lambda row: (row['id'], row['parent_id']))
rdd = rdd.groupWith(rdd)
rdd = rdd.flatMap(lambda kvw: [((v, w), 1) for v in kvw[1][0] for w in kvw[1][1]])
rdd.reduceByKey(operator.add)
print(rdd.count())
'''

def f(row):
    return [tok2idx.get(tok, 0) for tok in word_tokenize(row['body'])]

def g(row):
    return len(word_tokenize(row['body']))

df = ss.read.format('json').load(sys.argv[1])
rdd = df[['parent_id', 'body']].rdd
tok2idx = pickle.load(open(sys.argv[2], 'rb'))
rdd = rdd.map(f).persist()
print('idx')
np.save('idx', np.array(rdd.reduce(operator.add).collect(), dtype=np.int32))
print('seg')
np.save('seg', np.array(rdd.map(len).collect(), dtype=np.int32))
'''
rdd = rdd.flatMap(f)
rdd.saveAsTextFile(sys.argv[1] + '-part')
'''
