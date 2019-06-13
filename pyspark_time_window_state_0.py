import numpy as np

from pyspark.sql.session import sparksession

if __name__ == '__main__':
    ss = SparkSession.builder.getOrCreate()
    cmnt_df = ss.read.orc('cmnt-df')

    pid = np.array(cmnt_df.select("pid").rdd.flatMap(lambda x: x).collect())
    uid = np.array(cmnt_df.select("uid").rdd.flatMap(lambda x: x).collect())
    utc = np.array(cmnt_df.select("utc").rdd.flatMap(lambda x: x).collect())

    np.savetxt('pid', pid, '%d')
    np.savetxt('uid', uid, '%d')
    np.savetxt('utc', utc, '%d')
