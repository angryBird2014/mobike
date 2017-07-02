import pandas as pd
import numpy as np
import gc
import pickle

train_path = '../data/train.csv'
train_dump = '../dump/train.pkl'
test_path = '../data/test.csv'
test_dump = '../dump/test.pkl'
geohash_postion = '../dump/geohash_to_position_dict.pkl'

def get_data():

    geo_pos_dict = pickle.load(open(geohash_postion,'rb'))

    train = pd.read_csv(train_path)

    train['start_time'] = pd.to_datetime(train['starttime'])
    train['start_year'] = train['start_time'].apply(lambda x:x.year)
    train['start_month'] = train['start_time'].apply(lambda x:x.month)
    train['start_day'] = train['start_time'].apply(lambda x: x.day)
    train['start_hour'] = train['start_time'].apply(lambda x: x.hour)
    train['start_minute'] = train['start_time'].apply(lambda x: x.minute)
    train['start_second'] = train['start_time'].apply(lambda x: x.second)

    train.drop(['starttime'],axis=1,inplace=True)


    train['start_longitude'] = train['geohashed_start_loc'].apply(lambda x:geo_pos_dict[x][0])
    train['start_latitude'] = train['geohashed_start_loc'].apply(lambda x: geo_pos_dict[x][1])

    train['end_longitude'] = train['geohashed_end_loc'].apply(lambda x:geo_pos_dict[x][0])
    train['end_latitude'] = train['geohashed_end_loc'].apply(lambda x: geo_pos_dict[x][1])

    print(train.columns)
    pickle.dump(train,open(train_dump,'wb'),protocol=4)

    del train
    gc.collect()

    test = pd.read_csv(test_path)
    test['start_time'] = pd.to_datetime(test['starttime'])
    test['start_year'] = test['start_time'].apply(lambda x: x.year)
    test['start_month'] = test['start_time'].apply(lambda x: x.month)
    test['start_day'] = test['start_time'].apply(lambda x: x.day)
    test['start_hour'] = test['start_time'].apply(lambda x: x.hour)
    test['start_minute'] = test['start_time'].apply(lambda x: x.minute)
    test['start_second'] = test['start_time'].apply(lambda x: x.second)

    test.drop(['starttime'], axis=1, inplace=True)


    test['start_longitude'] = test['geohashed_start_loc'].apply(lambda x: geo_pos_dict[x][0])
    test['start_latitude'] = test['geohashed_start_loc'].apply(lambda x: geo_pos_dict[x][1])
    print(test.columns)

    pickle.dump(test,open(test_dump,'wb'),protocol=4)
    del test
    gc.collect()




if __name__ == '__main__':
    get_data()