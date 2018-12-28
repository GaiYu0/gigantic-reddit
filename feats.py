import pickle
import sys
from nltk.tokenize import word_tokenize
import numpy as np
from pyspark.sql.session import SparkSession
import scipy.sparse as sps

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

def indexize(row):
    return [tok2idx.get(tok, 0) for tok in word_tokenize(row['body'])]

tok2idx = pickle.load(open(sys.argv[2], 'rb'))
for f in sys.argv[1:]:
    df = ss.read.format('json').load(f)
    rdd = df[['id', 'body']].rdd
    rdd = rdd.map(indexize).persist()
    indices = np.array(rdd.flatMap(lambda x: x).collect(), dtype=np.int32)
    len_list = rdd.map(len).collect()
    len_array = np.expand_dims(np.array(len_list, dtype=np.int32), 1)
    indptr = np.cumsum(np.array([0] + len_list, dtype=np.int32))

embeddings = np.load('embeddings.npy')
nnz = len(indices)
row = len(indptr) - 1
col = len(embeddings)
csr = sps.csr_matrix((np.ones(nnz, dtype=np.int32), indices, indptr), shape=(row, col), dtype=np.int32)
x = csr.dot(embeddings) / len_array
np.save('x', x)
