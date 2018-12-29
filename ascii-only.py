import operator
import sys
from pyspark.sql.session import SparkSession

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel(logger.Level.WARN)
logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)

df = ss.read.format('json').load(sys.argv[1])
schema = [field.name for field in df.schema]
def english_only(row):
    for field in schema:
        return all(ord(x) < 128 for x in row[field])
print(df.rdd.map(english_only).reduce(operator.__and__))
