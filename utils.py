from inspect import currentframe, getframeinfo
import time
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

filename = lambda: getframeinfo(currentframe().f_back).filename
lineno = lambda: getframeinfo(currentframe().f_back).lineno

class Timer:
    def __init__(self, prefix):
        self.prefix = prefix

    def __enter__(self):
        self.t = time.time()
        print(self.prefix)

    def __exit__(self, *args):
        print(time.time() - self.t)

def getter(field):
    return lambda row: row[field]

int36 = lambda x: int(x, 36)
udf_int36 = udf(int36, IntegerType())
fst = lambda x: x[0]
snd = lambda x: x[1]
starzip = lambda x: zip(*x)
column2rdd = lambda column: column.rdd.flatMap(lambda x: x)
column2list = lambda column: column2rdd(column).collect()
