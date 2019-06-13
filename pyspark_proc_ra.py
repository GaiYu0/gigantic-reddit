import argparse

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--ra', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    ss = SparkSession.builder.getOrCreate()
    sc = ss.sparkContext
    user_df = ss.read.json(args.ra) \
                     .withColumnRenamed('id', 'uid') \
                     .withColumn('uid', regexp_replace('uid', 't2_', '')) \
                     .withColumn('uid', utils.udf_int36('uid')) \
                     .withColumnRenamed('name', 'username') \
                     .select('uid', 'username')
    u_df = user_df.groupBy('username').count().filter('count = 1').drop('count')
    user_df.join(u_df, on='username').toPandas().to_hdf('user-df', 'df')
#   user_df.join(u_df, on='username').write.orc('user-df', mode='overwrite')
