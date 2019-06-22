import dask
import numpy as np
from pyspark.sql.session import SparkSession

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext

# user_df = ss.read.orc('user-df')
# post_df = ss.read.orc('post-df')
cmnt_df = ss.read.orc('cmnt-df')
user_df = cmnt_df.select('uid').dropDuplicates()
post_df = cmnt_df.select('pid').dropDuplicates()

collect_column = lambda df, column: df.select(column).rdd.flatMap(lambda x: x).collect()
cmnt_df = cmnt_df.join(sc.parallelize(zip(collect_column(post_df, 'pid'), range(post_df.count()))).toDF(['pid', 'compact_pid']), 'pid') \
                 .join(sc.parallelize(zip(collect_column(user_df, 'uid'), range(user_df.count()))).toDF(['uid', 'compact_uid']), 'uid')

p2u = cmnt_df.rdd.map(lambda row: [row['compact_pid'], row['compact_uid']]).groupByKey().sortByKey()
p2u_indptr = np.cumsum([0] + p2u.map(lambda kv: len(kv[1])).collect())
p2u_indices = np.array(p2u.flatMap(lambda kv: kv[1]).collect())

u2p = cmnt_df.rdd.map(lambda row: [row['compact_uid'], row['compact_pid']]).groupByKey().sortByKey()
u2p_indptr = np.cumsum([0] + u2p.map(lambda kv: len(kv[1])).collect())
u2p_indices = np.array(u2p.flatMap(lambda kv: sorted(kv[1])).collect())

print('# posts:', len(p2u_indptr) - 1)
print('# users:', len(u2p_indptr) - 1)
print('# cmnts:', len(p2u_indices))
dask.compute(dask.delayed(np.savetxt)('p2u-indptr', p2u_indptr, fmt='%d'),
             dask.delayed(np.savetxt)('p2u-indices', p2u_indices, fmt='%d'),
             dask.delayed(np.savetxt)('u2p-indptr', u2p_indptr, fmt='%d'),
             dask.delayed(np.savetxt)('u2p-indices', u2p_indices, fmt='%d'))
