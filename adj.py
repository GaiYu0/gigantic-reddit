import sys
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
    df = ss.read.format('json').load(f)
    link_ids.extend(df[['link_id']].rdd.map())
    authors.extend(df[['authors']])

data = np.ones(len(ids), dtype=np.int8)
p2u = sps.coo_matrix((data, (link_ids, authors)))
u2p = sps.coo_matrix((data, (authors, link_ids)))
p2p = p2u.dot(u2p)
