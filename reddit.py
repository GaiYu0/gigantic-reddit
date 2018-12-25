import sys
import pyspark

def map_fn(row):
    d = row.asDict()
    return {d['parent_id'] : d['id']}

def reduce_fn(d, e):
    d.update(e)
    return d

df = spark.read.format('json').load(sys.argv[1])
conf = SparkConf().setAppName('reddit').setMaster('local')
sc = SparkContext(conf=conf)
rdd = sc.distribute(df[['id', 'parent_id']].collect())
csr = rdd.map(map_fn).reduce(reduce_fn)
