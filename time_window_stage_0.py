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
    user_df_indexed_by_username = ddf.read_hdf('user-df/indexed-by-username-*', '/df')

    cmnt_bag = db.read_text(args.rc, blocksize=args.bs).map(json.loads)
    cmnt_df = cmnt_bag.map(lambda d: {'pid' : utils.int36(d['link_id'].replace('t3_', '')),
                                      'username' : d['author'].encode('utf-8'),
                                      'utc' : int(d['created_utc'])}).to_dataframe()
    # cmnt_df = cmnt_df.set_index(cmnt_df.username)
    '''
    cmnt_df = cmnt_df.merge(user_df_indexed_by_username,
                            how='inner', on='username', left_index=True, right_index=True)
    '''
    cmnt_df = cmnt_df.merge(user_df_indexed_by_username,
                            how='inner', on='username')

    pid = np.array(cmnt_df.pid)
    uid = np.array(cmnt_df.uid)
    utc = np.array(cmnt_df.utc)

    print(len(pid))
    dask.compute(dask.delayed(np.savetxt)('pid', pid, '%d'),
                 dask.delayed(np.savetxt)('uid', uid, '%d'),
                 dask.delayed(np.savetxt)('utc', utc, '%d'))
