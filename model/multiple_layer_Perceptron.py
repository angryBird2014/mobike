import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import logging


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("bn_epsilon", 1e-5, "The epsilon of batch normalization")
flags.DEFINE_integer('batch_size',256,'batch size')
flags.DEFINE_float('dropout_keep_prob',0.5,'drop_out_rate')
flags.DEFINE_string('summaries_dir','../summary','summaries_dir')
flags.DEFINE_string('checkpoint','../summary/checkpoint','check_point')

def full_connect(inputs,weight_shape,biase_shape,layer_name):

    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = tf.Variable(tf.truncated_normal(weight_shape,stddev = 0.01,dtype=tf.float64))

            tf.summary.histogram('weight',weights)
        with tf.name_scope('biaes'):
            biases = tf.Variable(tf.truncated_normal([biase_shape,],stddev=0.01,dtype=tf.float64))
            tf.summary.histogram('biaes', biases)
        with tf.name_scope(layer_name+'_output'):
            layer = tf.matmul(inputs,weights) + biases
        with tf.name_scope(layer_name + '_output_dropout'):
            layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)
            tf.summary.histogram('dropout_output', layer)

        with tf.name_scope(layer_name + 'output_relu'):
            layer = tf.nn.relu(layer)
            tf.summary.histogram('relu_output', layer)

        with tf.name_scope(layer_name + 'batch_normalize'):
            mean, var = tf.nn.moments(layer, axes=[0])
            scale = tf.Variable(tf.truncated_normal([biase_shape,],stddev=0.01,dtype=tf.float64))

            shift = tf.Variable(tf.truncated_normal([biase_shape,],stddev=0.01,dtype=tf.float64))

            layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                              FLAGS.bn_epsilon)
            tf.summary.histogram('batch_normalizer', layer)


    return layer


def construct_graph(x,center_point):

    '''
    
    :param x: input tensor ,shape(n_samples,input_dimension)
    :param center_point: median_shift point
    :return: 
    '''


    hidden_layer1_number = 256
    hidden_layer1 = full_connect(x,[x.shape[1].value,hidden_layer1_number],hidden_layer1_number,"layer1")

    hidden_layer2_number = len(center_point)
    hidden_layer2 = full_connect(hidden_layer1,[hidden_layer1.shape[1].value,hidden_layer2_number],hidden_layer2_number,"layer2")
    with tf.name_scope('output_layer'):
        with tf.name_scope('softmax'):
            softmax_layer = tf.nn.softmax(hidden_layer2)
            tf.summary.histogram('softmax',softmax_layer)
        with tf.name_scope('output_weight'):
            weight = np.array(center_point)  #最后一层权重是常量,不更新,其值是center_point

            weight_layers = tf.constant(value=weight,dtype=tf.float64,shape=weight.shape)
            tf.summary.histogram('softmax', weight_layers)
        #bias_layer3 = tf.Variable(tf.truncated_normal([1],stddev=0.1,dtype=tf.float64))

        #output = tf.matmul(softmax_layer,weight_layers) + bias_layer3 #element_wise muitiply
        with tf.name_scope('output'):
            output = tf.matmul(softmax_layer, weight_layers)  # element_wise muitiply
            tf.summary.histogram('output', output)


    return output

def generate_random_batch(data,label,batch_size):

    indics = np.random.randint(0,len(data),size=batch_size)
    data_batch = data.iloc[indics]
    label_batch = label.iloc[indics]
    return data_batch.as_matrix(),label_batch.as_matrix()


if __name__ == '__main__':

    train = pickle.load(open('../dump/train.pkl','rb'))

    train = train.drop(['start_time','geohashed_start_loc','geohashed_end_loc'],axis = 1)

    train = train.astype(np.float64)
    train_label = train[['end_longitude','end_latitude']]

    train_data = train.drop(['end_longitude','end_latitude'],axis = 1)

    center_point = pickle.load(open('../dump/cluster_center.pkl','rb'))



    input_dimension = len(train_data.columns)
    output_dimension = 2
    x = tf.placeholder(tf.float64,[None,input_dimension])
    y = tf.placeholder(tf.float64,[None,output_dimension])
    output = construct_graph(x,center_point)
    with tf.name_scope('loss'):
        loss = tf.reduce_sum(tf.add(tf.square(tf.sin((output[:,1]-y[:,1])/2)),tf.multiply(tf.cos(y[:,1]),tf.multiply(tf.square(tf.sin((output[:,0]-y[:,0])/2)),tf.cos(output[:,1])))))
        tf.summary.scalar('loss',loss)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    #正确性咋衡量,毕竟差一个小数位就是错的,只能看Loss?




    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)
        saver = tf.train.Saver()
        #test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

        for i in range(200000):

            train,label = generate_random_batch(train_data,train_label,FLAGS.batch_size)
            if i % 100 == 0:
                # save the model
                loss_ = sess.run([loss],feed_dict={x:train,y:label})
                print('loss',loss_)
            if i % 3000 == 0:
                saver.save(sess, FLAGS.checkpoint+'model')
            _,summary = sess.run([train_step,merged],feed_dict={x: train, y: label})
            train_writer.add_summary(summary,i)





