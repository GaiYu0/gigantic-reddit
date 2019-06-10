import argparse
import dask.bag as db
import dask.dataframe as ddf
from dask.distributed import Client
import json
import numpy as np
import pandas as pd
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--rs', type=str)

if __name__ == '__main__':
    client = Client()

    args = parser.parse_args()

    post_bag = db.read_text(args.rs, blocksize=args.bs).map(json.loads)
    post_df = post_bag.map(lambda d: {'pid' : d['id'],
                                                    'title' : d['title'],
                                                    'username' : d['author'].encode('utf-8')}).to_dataframe()
    post_df = post_df.set_index(post_df.username)
    user_df_indexed_by_username = ddf.read_hdf('user-df/indexed-by-username-*', '/df')
    post_df = post_df.merge(user_df_indexed_by_username, how='inner', on='username',
                            left_index=True, right_index=True)

    u = np.loadtxt('u', dtype=np.int64)
    v = np.loadtxt('v', dtype=np.int64)
    w = np.loadtxt('w', dtype=np.int64)
    pid = np.array(post_df.pid)
    isin = u.isin(pid) & v.isin(pid)
    u, v, w = dask.compute(dask.delayed(u.__getitem__)(isin),
                           dask.delayed(v.__getitem__)(isin),
                           dask.delayed(w.__getitem__)(isin))
    def pid2nid(x):
        unique, unique_inverse = np.unique(x, return_inverse=True)
        nid = np.arange(len(unique))
        return unique, nid, nid[unique_inverse]
    [unique, nid, src], [unique, nid, dst] = dask.compute(dask.delayed(pid2nid)(u),
                                                          dask.delayed(pid2nid)(v))
    pid_nid_df = ddf.from_pandas(pd.DataFrame({'pid' : unique, 'nid' : nid}))
    pid_nid_df = pid_nid_df.set_index(pid_nid_df.pid)
    post_df = post_df.drop('username')
    post_df = post_df.set_index(post_df.pid)
    pid_nid_title_df = pid_nid_df.merge(post_df, how='inner', on='pid',
                                        left_index=True, right_index=True)
    nid_title_df = pid_nid_title_df.drop('pid')
    # sort?

    def words2indices(row):
        d = defaultdict(lambda: 0)
        for tok in tokenizer(row):
            idx = tok2idx.get(tok, 0)  # TODO default value
            d[idx] += 1
        if d:
            n = sum(d.values())
            return (len(d), tuple((k, d[k] / n) for k in sorted(d)))
        else:
            return (0, tuple())

    tok2idx = pickle.load(open(sys.argv[1], 'rb'))
    embeddings = mx.nd.array(np.load(sys.argv[2]))
    bag = nid_title_df.to_bag()
    bag = bag.map(words2indices).persist()
    indptr = mx.nd.array(np.cumsum([0] + bag.map(utils.fst).compute()))
    bag = bag.map(utils.snd).flatten().persist()
    indices = mx.nd.array(bag.map(utils.fst).compute())
    data = mx.nd.array(bag.map(utils.snd).collect())
    shape = [len(indptr) - 1, len(embeddings)]
    csr_matrix = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
    x = mx.nd.sparse.dot(csr_matrix, embeddings).asnumpy()

    np.save('x', x)
    np.save('u', u)
    np.save('v', v)
    np.save('w', w)
