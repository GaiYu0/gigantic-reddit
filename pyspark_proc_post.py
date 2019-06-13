import argparse

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--rs', type=str, nargs='+')
parser.add_argument('--top-k', type=int)

args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
user_df = ss.read.orc('user-df')
post_df = None
for f in args.rs:
    df = ss.read.json(f)
    post_df = df if post_df is None else post_df.union(df)
post_df = post_df.withColumn('pid', utils.udf_int36('id')) \
                 .withColumnRenamed('author', 'username') \
                 .withColumn('subreddit', utils.udf_int36(regexp_replace('subreddit_id', 't5_', ''))) \
                 .select('pid', 'username', 'subreddit', 'title')
top_k = utils.fst(zip(*sorted(post_df.groupBy('subreddit').count().rdd.map(lambda x: [x.subreddit, x.count]).collect(), key=utils.snd)[:args.top_k]))
post_df.filter(post_df.subreddit.isin(topk))
       .join(user_df, ['username'])
       .drop('username')
       .write.orc('post-df')
