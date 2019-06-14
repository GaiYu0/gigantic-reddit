# TODO(yu): relabel subreddit
import argparse

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--rs', type=str, nargs='+')
parser.add_argument('--hi', type=str)
parser.add_argument('--lo', type=str)

args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
user_df = ss.read.orc('user-df')
post_df = None
for f in args.rs:
    df = ss.read.json(f)
    post_df = df if post_df is None else post_df.union(df)
post_df = post_df.withColumn('pid', utils.udf_int36('id')) \
                 .withColumnRenamed('author', 'username') \
                 .dropna(subset=['subreddit_id']) \
                 .withColumn('srid', utils.udf_int36(regexp_replace('subreddit_id', 't5_', ''))) \
                 .select('pid', 'username', 'srid', 'title')
lo = eval(args.lo)
hi = eval(args.hi)
top_k = next(zip(*sorted(post_df.groupBy('srid').count().rdd.map(lambda x: [x['srid'], x['count']]).collect(), key=utils.snd, reverse=True)[lo : hi]))
post_df.filter(post_df.srid.isin(*top_k)) \
       .join(user_df, ['username']) \
       .drop('username') \
       .write.orc('post-df', mode='overwrite')
