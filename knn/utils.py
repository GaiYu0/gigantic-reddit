import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

filename = lambda: getframeinfo(currentframe().f_back).filename
lineno = lambda: getframeinfo(currentframe().f_back).lineno

int36 = lambda x: int(x, 36)
udf_int36 = udf(int36, IntegerType())
fst = lambda x: x[0]
snd = lambda x: x[1]
starzip = lambda x: zip(*x)
column2rdd = lambda column: column.rdd.flatMap(lambda x: x)
column2list = lambda column: column2rdd(column).collect()

def loadtxt(sc, fname, convert, dtype):
    return np.array(sc.textFile(fname, use_unicode=False).map(convert).collect(), dtype=dtype)
