import argparse

import numpy as np
from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--rs', type=str, nargs='+')

args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
user_df = ss.read.orc('user-df')
post_df = None
for f in args.rs:
    df = ss.read.json(f).select('id', 'author', 'subreddit_id', 'title')
    post_df = df if post_df is None else post_df.union(df)
post_df = post_df.withColumn('pid', utils.udf_int36('id')) \
                 .withColumnRenamed('author', 'username') \
                 .dropna(subset=['subreddit_id']) \
                 .withColumn('srid', utils.udf_int36(regexp_replace('subreddit_id', 't5_', ''))) \
                 .select('pid', 'username', 'srid', 'title')
srid = map(int, np.load('srid.npy'))
post_df.filter(post_df.srid.isin(*srid)) \
       .join(user_df, ['username']) \
       .drop('username') \
       .write.orc('post-df', mode='overwrite')
