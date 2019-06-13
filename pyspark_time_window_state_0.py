session = SparkSession.builder.getOrCreate()
context = session.sparkContext
logger = context._jvm.org.apache.log4j

