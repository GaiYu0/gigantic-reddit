import argparse
import dask.bag as db
import dask.dataframe as dd
from dask.distributed import Client
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ra', type=str)
parser.add_argument('--rc', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    client = Client()

    user_bag = db.read_text(args.ra).map(json.loads)
    uid_username_df = user_bag.map(lambda d: {'uid' : int(d['id'].replace('t2_', ''), 36),
                                              'username' : d['name']}).to_dataframe()

    cmnt_bag = db.read_text(args.rc).map(json.loads)
    pid_bag = cmnt_bag.map(lambda d: int(d['link_id'].replace('t3_', ''), 36))
    pid = np.array(pid_bag.compute())
    username_df = cmnt_bag.map(lambda d: {'username' : d['author']}).to_dataframe()
    uid_df = dd.merge(username_df, uid_username_df, 'inner', 'username')
    uid = np.array(uid_df.uid)
    utc_bag = cmnt_bag.map(lambda d: int(d['created_utc']))
    utc = np.array(utc_bag.compute())

    print(len(pid))
    np.savetxt('pid', pid, '%d')
    np.savetxt('uid', uid, '%d')
    np.savetxt('utc', utc, '%d')
