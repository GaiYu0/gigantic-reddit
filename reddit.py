import operator
import sys
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

ss = SparkSession.builder.master('local').appName('reddit').config('spark.ui.showConsoleProgress', True).getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

df = ss.read.format('json').load(sys.argv[1])
rdd = df[['id', 'parent_id']].rdd
rdd = rdd.map(lambda row: (row['id'], row['parent_id']))
rdd = rdd.groupWith(rdd)
rdd = rdd.flatMap(lambda kvw: [((v, w), 1) for v in kvw[1][0] for w in kvw[1][1]])
rdd.reduceByKey(operator.add)
print(rdd.count())
