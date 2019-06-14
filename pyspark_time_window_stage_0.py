from functools import partial
import multiprocessing as mp

import numpy as np

from pyspark.sql.session import SparkSession

ss = SparkSession.builder.getOrCreate()
cmnt_df = ss.read.orc('cmnt-df')

pid = np.array(cmnt_df.select("pid").rdd.flatMap(lambda x: x).collect())
uid = np.array(cmnt_df.select("uid").rdd.flatMap(lambda x: x).collect())
utc = np.array(cmnt_df.select("utc").rdd.flatMap(lambda x: x).collect())

print(len(pid))
with mp.Pool(3) as pool:
    pool.starmap(partial(np.savetxt, fmt='%d'), [['pid', pid], ['uid', uid], ['utc', utc]])
