import sys
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

def row2kv_pair(row):
    d = row.asDict()
    return (d['parent_id'], d['id'])

ss = SparkSession.builder.master("local").appName("reddit").getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

df = ss.read.format('json').load(sys.argv[1])
rdd = sc.parallelize(df[['id', 'parent_id']].collect())
kv_pairs = rdd.map(row2kv_pair)
csr = kv_pairs.groupByKey().collect()
print(type(csr))
print(len(csr))
print(csr[0])
print(list(csr[0][1]))
