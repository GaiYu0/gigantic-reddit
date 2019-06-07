import argparse
import dask
import dask.bag as db
import dask.dataframe as dd
from dask.distributed import Client
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--ra', type=str)
parser.add_argument('--rc', type=str)

if __name__ == '__main__':
    client = Client()

    args = parser.parse_args()
    user_bag = db.read_text(args.ra, blocksize=args.bs).map(json.loads)
    uid_username_df = user_bag.map(lambda d: {'uid' : int(d['id'].replace('t2_', ''), 36),
                                              'username' : d['name']}).to_dataframe()

    cmnt_bag = db.read_text(args.rc, blocksize=args.bs).map(json.loads)

    print('pid')
    pid_bag = cmnt_bag.map(lambda d: int(d['link_id'].replace('t3_', ''), 36))
    pid = np.array(pid_bag.compute())
    username_df = cmnt_bag.map(lambda d: {'username' : d['author']}).to_dataframe()
    username_df = username_df.set_index(username_df.username)
    uid_username_df = uid_username_df.set_index(uid_username_df.username)
    username_df = dd.merge(username_df, uid_username_df, 'inner', 'username', left_index=True, right_index=True)

    print('uid')
    uid = np.array(username_df.uid)

    print('utc')
    utc_bag = cmnt_bag.map(lambda d: int(d['created_utc']))
    utc = np.array(utc_bag.compute())

    print(len(pid))
    dask.compute(dask.delayed(np.savetxt)('pid', pid, '%d'),
                 dask.delayed(np.savetxt)('uid', uid, '%d'),
                 dask.delayed(np.savetxt)('utc', utc, '%d'))
