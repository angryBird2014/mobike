import pandas as pd
import geohash
import pickle
from functools import reduce
train_path = '../data/train.csv'
test_path = '../data/test.csv'



if __name__ == '__main__':

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    pos = set(train['geohashed_start_loc']).union(set(train['geohashed_end_loc'])).union(test['geohashed_start_loc'])

    geohash_position = {}

    position_geohash = {}

    for ele in pos:
        geohash_position[ele] = geohash.decode(ele)
        position_geohash[geohash.decode(ele)]=ele

    pickle.dump(geohash_position,open('../dump/geohash_to_position_dict.pkl','wb'),protocol=4)

    pickle.dump(position_geohash,open('../dump/position_to_geohash_dict.pkl','wb'),protocol=4)




