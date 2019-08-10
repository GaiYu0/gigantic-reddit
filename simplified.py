import argparse
from functools import partial

import numpy as np
from pyspark.sql.functions import regexp_replace, udf
from pyspark.sql.session import SparkSession
import pyspark.sql.types as IntegerType

int36 = udf(partial(int, base=36), types.IntegerType())

parser = argparse.ArgumentParser()
parser.add_argument('--rc', type=str, nargs='+')
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()

rs = None
for x in args.rs:
    df = ss.read.json(x).select('id').dropna('any')
    rs = df if rs is None else rs.union(df)

rs = rs.withColumnRenamed('id', 'pid')
rs = rs.withColumn('pid', regexp_replace(rs.pid))
rs = rs.withColumn('pid', int36(rs.pid))

rc = None
for x in args.rc:
    df = ss.read.json(x).select('author', 'created_utc', 'link_id').dropna('any')
    rc = df if rc is None else rc.union(df)

authors = rc.select('author').dropDuplicates().rdd.flatMap(id_map).collect()
authors.remove('[deleted]')
ra = sc.parallelize(zip(range(len(authors)), authors)).toDF('uid INT, author STRING')
rc = rc.join(ra, 'author')

rc = rc.withColumnRenamed('link_id', 'pid')
rc = rc.withColumn('pid', regexp_replace(rc.pid, 't3_', ''))
rc = rc.withColumn('pid', int36(rc.pid))
rc = rc.join(rs, 'pid')

rc = rc.withColumn('utc', rc.created_utc.cast(IntegerType()))

pid = np.array(rc.select('pid').rdd.flatMap(id_map).collect())
uid = np.array(rc.select('uid').rdd.flatMap(id_map).collect())
utc = np.array(rc.select('utc').rdd.flatMap(id_map).collect())

uniq, inv = np.unique(pid, return_inverse=True)
pid = np.arange(len(uniq))[inv]

np.save('pid', pid)
np.save('uid', uid)
np.save('utc', utc)
