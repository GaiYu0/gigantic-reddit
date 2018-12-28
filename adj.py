import sys
import numpy as np
from pyspark.sql.session import SparkSession
import scipy.sparse as sps

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

link_ids = []
authors = []
for f in sys.argv[1:]:
    df = ss.read.format('json').load(f).filter(df.author != '[deleted]')
    link_ids.extend(df.rdd.map(lambda row: int(row['link_id'][3:])).collect())
    authors.extend(df.rdd.map(lambda row: sum(ord(x) for x in row['author'])).collect())

data = np.ones(len(link_ids), dtype=np.int8)
p2u = sps.coo_matrix((data, (link_ids, authors)), dtype=np.int8)
u2p = sps.coo_matrix((data, (authors, link_ids)), dtype=np.int8)
p2p = p2u.dot(u2p)
print(p2p.shape, max(link_ids))
print(p2p.nnz)
