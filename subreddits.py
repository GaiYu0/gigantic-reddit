import sys
from pyspark.sql.session import SparkSession

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

for f in sys.argv[2:]:
    df = ss.read.format('json').load(f)
    df = df.groupBy('subreddit_id').count()
    topk = df.orderBy('count', ascending=False).limit(int(sys.argv[1])).collect()

print(topk)
