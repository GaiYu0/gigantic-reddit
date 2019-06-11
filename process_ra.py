import argparse
import dask.bag as db
from dask.distributed import Client
# import dask.multiprocessing
import json
import utils

# parser = argparse.ArgumentParser()
# parser.add_argument('--bs', type=int)
# parser.add_argument('--ra', type=str)
# args = parser.parse_args()

def main(args):
#   dask.config.set(scheduler='processes')
    client = Client()

    user_bag = db.read_text('RA_2005', blocksize=512 * 1024 * 1024).map(json.loads)
#   user_bag = db.read_text(args.ra, blocksize=args.bs).map(json.loads)
    '''
    user_bag = user_bag.map(lambda d: {'uid' : utils.int36(d['id'].replace('t2_', '')),
                                       'username' : d['name']})
    username_frequency_bag = user_bag.pluck('username').frequencies()
    username_frequency_df = username_frequency_bag.map(lambda kv: {'username' : utils.fst(kv), 'frequency' : utils.snd(kv)}).to_dataframe()
    '''
    user_df = user_bag.to_dataframe()
#   user_df = user_df.merge(username_frequency_df, on='username', how='inner')
    user_df.to_hdf('user-df/*', '/df', mode='w')
    '''
    user_df_indexed_by_uid = user_df.set_index(user_df.uid)
    user_df_indexed_by_username = user_df.set_index(user_df.username)
    user_df_indexed_by_uid.to_hdf('user-df/indexed-by-uid-*', '/df', mode='w')
    user_df_indexed_by_username.to_hdf('user-df/indexed-by-username-*', '/df', mode='w')
    '''

if __name__ == '__main__':
    main(None)
#   main(args)
