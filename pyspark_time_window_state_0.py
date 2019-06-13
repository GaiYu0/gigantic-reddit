import argparse

import numpy as np
from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession
from pyspark.sql.types import IntegerType

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--rc', type=str, nargs='+')

if __name__ == '__main__':
    args = parser.parse_args()

    ss = SparkSession.builder.getOrCreate()
    user_df = ss.read.orc('user-df')
    cmnt_df = None
    for f in args.rc:
        df = ss.read.json(f)
        cmnt_df = df if cmnt_df is None else cmnt_df.union(df)
    cmnt_df = cmnt_df.withColumnRenamed('link_id', 'pid') \
                     .withColumn('pid', regexp_replace('pid', 't3_', '')) \
                     .withColumn('pid', utils.udf_int36('pid')) \
                     .withColumnRenamed('author', 'username') \
                     .withColumn('utc', cmnt_df.created_utc.cast(IntegerType())) \
                     .join(user_df, ['username'])

    pid = np.array(cmnt_df.select("pid").rdd.flatMap(lambda x: x).collect())
    uid = np.array(cmnt_df.select("uid").rdd.flatMap(lambda x: x).collect())
    utc = np.array(cmnt_df.select("utc").rdd.flatMap(lambda x: x).collect())
