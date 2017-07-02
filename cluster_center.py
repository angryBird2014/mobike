from sklearn import cluster
import pickle
import numpy as np

def get_set_value(set):
    value = list(set)
    return [value[0][0],value[0][1]]

def get_position():

    position = pickle.load(open('dump/geohash_to_position_dict.pkl','rb'))

    position_set = position.values()

    position_array = [list(pos) for pos in position_set]


    position_matrix = np.array(position_array)

    banwidth = cluster.estimate_bandwidth(position_matrix,quantile=0.3,n_jobs=-1)

    ms = cluster.MeanShift(bandwidth=banwidth,bin_seeding=False,n_jobs=-1)

    ms.fit(position_matrix)

    cluster_center = ms.cluster_centers_

    print(cluster_center)
    pickle.dump(cluster_center,open('dump/cluster_center.pkl','wb'),protocol=4)

if __name__ == '__main__':
    get_position()