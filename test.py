import pickle
import pandas as pd
import numpy as np

def read_position():

    data = pickle.load(open('dump/position.pkl','rb'))
    print(data.head)

def read_center():
    data = pickle.load(open('dump/cluster_center.pkl','rb'))
    hash = pickle.load(open('dump/position_to_geohash.pkl','rb'))
    print(data)
    print(len(data))

def getmax_discert():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    print(np.max())

def get_data():
    data = pickle.load(open('dump/train.pkl','rb'))
    print(len(data.columns))

def test_zip():
    a = [[1,2,3],[2,3,4]]
    b= [2,3,4]
    data = []
    for x,y in zip(a,b):
        data.append(x+[y])
    print(data)
    pass
if __name__ == '__main__':
    #read_position()
    #read_center()
    #get_data()
    test_zip()
    pass