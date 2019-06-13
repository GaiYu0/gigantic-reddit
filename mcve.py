import dask.bag as db
from dask.distributed import Client
import json

def main():
    client = Client()
    user_bag = db.read_text('RA_2005', blocksize=512 * 1024 * 1024).map(json.loads)
    user_df = user_bag.to_dataframe()
    user_df.to_hdf('user-df/*', '/df', mode='w')

if __name__ == '__main__':
    main()
