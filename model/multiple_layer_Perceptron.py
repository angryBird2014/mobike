import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import logging

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("bn_epsilon", 1e-5, "The epsilon of batch normalization")

def full_connect(inputs,weight_shape,biase_shape,layer_number):


    weights = tf.Variable(('weight_layer'+str(layer_number)),weight_shape,
                          initial_value=tf.random_normal(weight_shape,stddev = 0.01))
    biases = tf.get_variable(("biases"+str(layer_number)),
                             biase_shape,
                             initializer=tf.random_normal(biase_shape,stddev=0.01))
    layer = tf.multiply(inputs,weights) + biases

    mean, var = tf.nn.moments(layer, axes=[0])
    scale = tf.get_variable("scale",
                            biase_shape,
                            initializer=tf.random_normal_initializer())
    shift = tf.get_variable("shift",
                            biase_shape,
                            initializer=tf.random_normal_initializer())
    layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                      FLAGS.bn_epsilon)
    return layer


def construct_graph():
    pass

if __name__ == '__main__':

    train = pickle.load(open('dump/train.pkl','rb'))
    input_dimension = len(train.columns)
    output_dimension = 2
    x = tf.placeholder(tf.float64,[None,input_dimension])
    y = tf.placeholder(tf.float64,[None,output_dimension])
    construct_graph()

    pass