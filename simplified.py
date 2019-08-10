import argparse
from functools import partial
import pickle

import numpy as np
from pyspark.sql.functions import regexp_replace, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

id_map = lambda x: x
int36 = udf(partial(int, base=36), IntegerType())

parser = argparse.ArgumentParser()
parser.add_argument('--rc', type=str, nargs='+')
parser.add_argument('--rs', type=str, nargs='+')
parser.add_argument('--subreddits', type=str)
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext

rs = None
for x in args.rs:
    df = ss.read.json(x).select('id', 'subreddit_id')
    rs = df if rs is None else rs.union(df)
rs = rs.dropna('any')

rc = None
for x in args.rc:
    df = ss.read.json(x).select('author', 'created_utc', 'link_id')
    rc = df if rc is None else rc.union(df)
rc = rc.dropna('any')

rs = rs.withColumnRenamed('subreddit_id', 'srid')
rs = rs.filter(rs.srid.isin(pickle.load(open(args.subreddits, 'rb'))))
rs = rs.withColumn('srid', regexp_replace(rs.srid, 't5_', ''))
rs = rs.withColumn('srid', int36(rs.srid))
srid = np.array(rs.select('srid').rdd.flatMap(id_map).collect())
uniq, inv = np.unique(srid, return_inverse=True)
np.save('srid', np.arange(len(uniq))[inv])

rc = rc.withColumnRenamed('link_id', 'link_id')
rc = rc.withColumn('link_id', regexp_replace(rc.link_id, 't3_', ''))
rc = rc.withColumn('link_id', int36(rc.link_id))

rs = rs.withColumn('id', int36(rs.id))
link_ids = rs.select('id').rdd.flatMap(id_map).collect()
df = sc.parallelize(zip(range(len(link_ids)), link_ids)).toDF('pid INT, link_id INT')
rc = rc.join(df, 'link_id')

authors = rc.select('author').dropDuplicates().rdd.flatMap(id_map).collect()
authors.remove('[deleted]')
ra = sc.parallelize(zip(range(len(authors)), authors)).toDF('uid INT, author STRING')
rc = rc.join(ra, 'author')

rc = rc.withColumn('utc', rc.created_utc.cast(IntegerType()))

pid = np.array(rc.select('pid').rdd.flatMap(id_map).collect())
uid = np.array(rc.select('uid').rdd.flatMap(id_map).collect())
utc = np.array(rc.select('utc').rdd.flatMap(id_map).collect())

np.save('pid', pid)
np.save('uid', uid)
np.save('utc', utc)
