import argparse

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--rc', type=str, nargs='+')

args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
user_df = ss.read.orc('user-df')
post_df = ss.read.orc('post-df')

cmnt_df = None
for f in args.rc:
    df = ss.read.json(f)
    cmnt_df = df if cmnt_df is None else cmnt_df.union(df)
cmnt_df.withColumnRenamed('link_id', 'pid') \
       .withColumn('pid', regexp_replace('pid', 't3_', '')) \
       .withColumn('pid', utils.udf_int36('pid')) \
       .withColumnRenamed('author', 'username') \
       .withColumn('utc', cmnt_df.created_utc.cast(IntegerType())) \
       .select('pid', 'username', 'utc') \
       .join(post_df.drop('uid', 'srid', 'title'), ['pid']) \
       .join(user_df, ['username']) \
       .drop('username') \
       .write.orc('cmnt-df', mode='overwrite')
