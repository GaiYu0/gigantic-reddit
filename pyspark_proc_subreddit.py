import argparse
import pickle

from pyspark.sql.functions import regexp_replace
from pyspark.sql.session import SparkSession

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--rx', type=str, nargs='+')
parser.add_argument('--hi', type=str)
parser.add_argument('--lo', type=str)
args = parser.parse_args()

lo = eval(args.lo)
hi = eval(args.hi)
ss = SparkSession.builder.getOrCreate()
df = None
for f in args.rx:
    df = ss.read.json(f)
    df = df if df is None else df.union(df)
subreddits = next(zip(*sorted(df.groupBy('srid').count().rdd.map(lambda x: [x['srid'], x['count']]).collect(), key=utils.snd, reverse=True)[lo : hi]))
df = df.dropna(subset=['subreddit_id']) \
       .withColumn('srid', utils.udf_int36(regexp_replace('subreddit_id', 't5_', ''))) \
       .groupBy('srid').count().toPandas()
srid_df = df.sort_values(by='count', ascending=False)[lo : hi]
srid = list(srid_df['srid'])
pickle.dump(srid, 'srid')
