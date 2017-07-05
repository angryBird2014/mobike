import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.preprocessing import OneHotEncoder




R = 6.3781e3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("bn_epsilon", 1e-5, "The epsilon of batch normalization")
flags.DEFINE_integer('batch_size',256,'batch size')

flags.DEFINE_string('summaries_dir','../summary','summaries_dir')
flags.DEFINE_string('checkpoint','../summary/checkpoint','check_point')


def get_train_data():
    train = pickle.load(open('../dump/train.pkl', 'rb'))

    train = train.drop(['start_time', 'geohashed_start_loc', 'geohashed_end_loc'], axis=1)

    train = train.astype(np.float64)
    train_label = train[['end_longitude', 'end_latitude']]

    train_data = train.drop(['end_longitude', 'end_latitude'], axis=1)

    return train_data,train_label

def get_test_data():
    test = pickle.load(open('../dump/test.pkl','rb'))

    test = test.drop(['start_time','geohashed_start_loc'],axis = 1)

    test = test.astype(np.float64)

    return test.as_matrix()



def generate_train_random_batch(data,label,batch_size,is_train = True):


    indics = np.random.randint(0,len(data),size=batch_size)
    vector = np.zeros([len(data)])
    vector[indics] = 1

    #y_label = OneHotEncoder(len(data),indics,sparse=False)

    data_batch = data.iloc[indics]
    if is_train:
        label_batch = label.iloc[indics]
        return data_batch.as_matrix(),label_batch.as_matrix(),vector
    else:
        return data_batch.as_matrix(),vector

if __name__ == '__main__':

    train_data, train_label = get_train_data()

    test_data = get_test_data()

    position = pickle.load(open('../dump/geohash_to_position_dict.pkl', 'rb'))
    position_value = list(position.values())

    input_dimension = len(train_data.columns)
    output_dimension = 2
    x = tf.placeholder(tf.float32, [None, input_dimension])
    y = tf.placeholder(tf.float32, [None, output_dimension])

    #layer1
    hidden_layer1_num = 216
    hidden_layer1 = tf.contrib.layers.fully_connected(x,hidden_layer1_num,activation_fn=None)
    hidden_layer1 = tf.layers.batch_normalization(hidden_layer1,center=True,scale = True)
    hidden_layer1 = tf.nn.relu(hidden_layer1,'layer2_relu')


    #layer2
    hidden_layer2_num = 512
    hidden_layer2 = tf.contrib.layers.fully_connected(hidden_layer1,hidden_layer2_num,activation_fn=None)
    hidden_layer2 = tf.layers.batch_normalization(hidden_layer2)
    hidden_layer2 = tf.nn.relu(hidden_layer2,'layer2_relu')


    #output_layer
    output_number = len(position_value)
    output_layer = tf.contrib.layers.fully_connected(hidden_layer2,output_number,activation_fn=None)
    output_layer = tf.nn.softmax(output_layer)
    #取出batch中的三个最大概率
    values,index = tf.nn.top_k(output_layer,3)



    #loss
    '''
    values(batch,3,2),y(batch,2)
    '''
    #loss = tf.reduce_sum(tf.multiply(R,tf.sqrt(tf.add(tf.pow(values[:,1]-y[:,1],2),tf.pow(tf.multiply((values[:,0]-y[:,0]),tf.cos((values[:,1]-y[:,1])/2)),2)))))
    loss = tf.re

    #train
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    #tensorflow session

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        for i in range(10000):

            train, label ,one_hot_label= generate_train_random_batch(train_data, train_label, FLAGS.batch_size)

            if i % 500 == 0:

                loss_,values_,index_ = sess.run([loss,values,index], feed_dict={x: train, y: label})
                print('loss', loss_,'values',values_,'index',index_)


            sess.run([train_step], feed_dict={x: train, y: label})









