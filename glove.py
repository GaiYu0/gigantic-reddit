import csv
import pickle
import sys
import numpy as np
import pandas as pds

frame = pds.read_table(sys.argv[1], header=None, index_col=0, keep_default_na=False, quoting=csv.QUOTE_NONE, sep=" ")
tok = frame.index
tok2idx = dict(zip(tok, range(len(tok))))
pickle.dump(tok2idx, open('tok2idx', 'wb'))
np.save('embeddings', frame.values.astype(np.float32))
