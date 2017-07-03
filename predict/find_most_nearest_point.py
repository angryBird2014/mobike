import pickle
import numpy as np
import pandas as pd


earth_radius = 6.3781e6
def get_distance(y_point,predict_point):
    '''
    
    :param y_point: 字典中的其他坐标(longtitude,latitude)
    :param predict_point: NN输出的一个坐标(longtitude,latitude) 
    :return: 两点之间的Haversine distance(scala)
    '''
    #distance = np.power(np.sin((predict_point[1]-y_point[1])/2),2) + np.cos(y_point[1]) * np.cos(predict_point[1]) * np.power(np.sin((predict_point[0]-y_point[0])/2),2)
    distance = earth_radius * np.sqrt(np.power((predict_point[0]-y_point[0])*np.cos((predict_point[1]-predict_point[1])/2),2)+np.power((predict_point[1]-y_point[1]),2))
    return distance


if __name__ == '__main__':
    test = pickle.load(open('../dump/test.pkl','rb'))
    orderID = test['orderid'].astype(np.str).tolist()
    predict = pickle.load(open('../dump/predict.pkl','rb'))[0]
    print(predict)

    greohash_point = pickle.load(open('../dump/geohash_to_position_dict.pkl','rb'))
    point_geohash = pickle.load(open('../dump/position_to_geohash_dict.pkl','rb'))
    greohash_point_value = list(greohash_point.values())
    final_output = []

    for point in predict:  #计算每个predict_point到列表中所有元素的距离,选择其中距离最小的3个

        distance_point = [ get_distance(x,y) for x,y in zip(greohash_point_value,[point.tolist()]*len(greohash_point_value))]
        min_three_point_index = np.array(distance_point).argsort()[:3]


        output = [point_geohash[greohash_point_value[index]] for index in min_three_point_index]
        print(output)
        #min_data_point = greohash_point_value[min_three_point_index]  # 最小的经纬度[(),(),()]
        #output = [point_geohash(point) for point in min_data_point]
        final_output.append(output)

    data = []
    for x,y in zip(orderID,final_output):
        data.append([x]+ y)
    with open('./submission.csv','w') as file:
        for item in data:
            line = ','.join(item)
            file.writelines(line + "\n")








