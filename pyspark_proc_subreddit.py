import argparse
import pickle

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--rx', type=str, nargs='+')
args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
df = None
for f in args.rx:
    x = ss.read.json(f).select('subreddit_id')
    df = x if df is None else df.union(x)
df = df.dropna(subset=['subreddit_id']) \
       .withColumn('srid', utils.udf_int36(regexp_replace('subreddit_id', 't5_', ''))) \
       .groupBy('srid').count().toPandas()
df.sort_values(by='count', ascending=False).reset_index().to_hdf('srid-df', '/df')
