import argparse
import logging

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--ra', type=str)

args = parser.parse_args()

ss = SparkSession.builder.getOrCreate()
user_df = ss.read.json(args.ra) \
                 .select('id', 'name') \
                 .withColumn('uid', utils.udf_int36(regexp_replace('id', 't2_', ''))) \
                 .withColumnRenamed('name', 'username')
u_df = user_df.groupBy('username').count().filter('count = 1').drop('count')
user_df.join(u_df, ['username']).write.orc('user-df', mode='overwrite')
