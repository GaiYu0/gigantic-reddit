import argparse
import dask.bag as db
from dask.distributed import Client, progress
import json

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--ra', type=str)

if __name__ == '__main__':
    client = Client()

    args = parser.parse_args()
    user_bag = db.read_text(args.ra, blocksize=args.bs).map(json.loads)
    user_bag = user_bag.map(lambda d: {'uid' : int(d['id'].replace('t2_', ''), 36),
                                       'username' : d['name']})
    user_bag = user_bag.groupby(lambda d: d['username'])
    user_bag = user_bag.map(lambda kv: [] if len(kv[1]) > 1 else kv[1])
    user_df = user_bag.flatten().to_dataframe()
    user_df_indexed_by_uid = user_df.set_index(user_df.uid)
    user_df_indexed_by_username = user_df.set_index(user_df.username)
    progress(user_df_indexed_by_uid)
    progress(user_df_indexed_by_username)
    user_df_indexed_by_uid.to_hdf('user_df/indexed_by_uid-*', '/df', mode='w')
    user_df_indexed_by_username.to_hdf('user_df/indexed_by_username-*', '/df', mode='w')
