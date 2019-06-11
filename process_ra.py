import argparse
import dask.bag as db
from dask.distributed import Client
# import dask.multiprocessing
import json
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--ra', type=str)
args = parser.parse_args()

def main(args):
#   dask.config.set(scheduler='processes')
    client = Client()

    user_bag = db.read_text(args.ra, blocksize=args.bs).map(json.loads)
    user_bag = user_bag.map(lambda d: {'uid' : utils.int36(d['id'].replace('t2_', '')),
                                       'username' : d['name']})
    user_bag = user_bag.groupby(lambda d: d['username'])
    user_bag = user_bag.map(lambda kv: [] if len(utils.snd(kv)) > 1 else utils.snd(kv))
    user_df = user_bag.flatten().to_dataframe()
    user_df_indexed_by_uid = user_df.set_index(user_df.uid)
    user_df_indexed_by_username = user_df.set_index(user_df.username)
    user_df_indexed_by_uid.to_hdf('user-df/indexed-by-uid-*', '/df', mode='w')
    user_df_indexed_by_username.to_hdf('user-df/indexed-by-username-*', '/df', mode='w')

if __name__ == '__main__':
    main(args)
