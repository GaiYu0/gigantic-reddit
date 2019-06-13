import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ra', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    ss = SparkSession.builder.getOrCreate()
    sc = ss.sparkContext
    user_df = ss.read.format('json').load(args.ra) \
                                    .withColumnRenamed('id', 'uid') \
                                    .withColumn('uid', regexp_replace('uid', 't2_', '')) \
                                    .withColumn('uid', utils.udf_int36('uid')) \
                                    .withColumnRenamed('name', 'username') \
                                    .select('uid', 'username')
                                    .groupBy('username') \
                                    .count() \
                                    .filter(user_df.count == 1) \
