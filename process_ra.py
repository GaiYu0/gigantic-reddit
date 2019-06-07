import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--ra', type=str)

if __name__ == '__main__':
    client = Client()

    args = parser.parse_args()
    user_bag = db.read_text(args.ra, blocksize=args.bs).map(json.loads)
    user_df = user_bag.map(lambda d: {'uid' : int(d['id'].replace('t2_', ''), 36),
                                      'username' : d['name']}).to_dataframe()
    user_df = user_df.drop_duplicates(subset=['username'], keep=False).persist()
    user_df_indexed_by_uid = user_df.set_index(user_df.uid)
    user_df_indexed_by_username = user_df.set_index(user_df.username)
    user_df_indexed_by_uid.to_hdf('user_df_indexed_by_uid-*', '/df', mode='w')
    user_df_indexed_by_username.to_hdf('user_df_indexed_by_username-*', '/df', mode='w')
