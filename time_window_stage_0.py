import argparse
import dask
import dask.bag as db
import dask.dataframe as ddf
from dask.distributed import Client
import json
import numpy as np
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--rc', type=str)
parser.add_argument('--rs', type=str)

if __name__ == '__main__':
    client = Client()

    args = parser.parse_args()

    # user_df_indexed_by_uid = ddf.read_hdf('user-df/indexed-by-uid-*')
    user_df_indexed_by_username = ddf.read_hdf('user-df/indexed-by-username-*')

    cmnt_bag = db.read_text(args.rc, blocksize=args.bs).map(json.loads)

    pid_bag = cmnt_bag.map(lambda d: utils.int36(d['link_id'].replace('t3_', '')))
    pid = np.array(pid_bag.compute())

    username_df = cmnt_bag.map(lambda d: {'username' : d['author']}).to_dataframe()
    username_df = username_df.set_index(username_df.username)
    username_df = ddf.merge(username_df, user_df_indexed_by_username,
                            how='inner', on='username', left_index=True, right_index=True)
    uid = np.array(username_df.uid)

    utc_bag = cmnt_bag.map(lambda d: int(d['created_utc']))
    utc = np.array(utc_bag.compute())

    print(len(pid))
    dask.compute(dask.delayed(np.savetxt)('pid', pid, '%d'),
                 dask.delayed(np.savetxt)('uid', uid, '%d'),
                 dask.delayed(np.savetxt)('utc', utc, '%d'))
