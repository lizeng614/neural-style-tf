#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:15:33 2017

@author: li
"""

import tensorflow as tf
import numpy as np

import vgg

# define global variable

CONTENT_LAYER = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

# define inference 
def inference(content,style,network):
    c_shape = (1,)+content.shape
    s_shape = (1,)+style.shape
    r_content = []
    r_styles = {}

    g = tf.Graph()
    with g.as_default(),g.device('/cpu:0'),tf.Session():
        p_content = tf.placeholder(tf.float32,shape=c_shape)
        c_net,mean_pixel = vgg.net(network,p_content)
        content_pre = np.array([vgg.preprocess(content,mean_pixel)])
        r_content = c_net[CONTENT_LAYER].eval(feed_dict={p_content: content_pre})
    
    g = tf.Graph()
    with g.as_default(),g.device('/cpu:0'),tf.Session():
        p_style = tf.placeholder(tf.float32,shape=s_shape)
        s_net, _ = vgg.net(network,p_style)
        style_pre = np.array([vgg.preprocess(style,mean_pixel)]) 
        for layer in STYLE_LAYERS:
            r_style = s_net[layer].eval(feed_dict={p_style:style_pre})
            r_style = np.reshape(r_style,(-1,r_style.shape[-1]))
            # calc the gram and normalize by the size of style mask
            r_styles[layer] = np.dot(r_style.T,r_style)/r_style.size
        
    # create a trainable image
    with tf.device('/cpu:0'):
        image = tf.get_variable(name='image',shape = c_shape,dtype='float',
                            initializer=tf.truncated_normal_initializer())
        net, _ = vgg.net(network,image)
    return r_content,r_styles,net,mean_pixel

# define loss
def loss(r_content,r_styles,net,content_weight,style_weight):
    # calc the content loss
    with tf.device('/cpu:0'),tf.Session():
        loss_content = tf.nn.l2_loss(net[CONTENT_LAYER]-r_content)
    # calc the style loss
        loss_style = 0
        for layer in r_styles:
            _, height, width,filters= map(lambda i: i.value, net[layer].get_shape())
            size = height*width*filters
            gram_net = tf.reshape(net[layer],(-1,filters))
            gram_net = tf.matmul(tf.transpose(gram_net),gram_net)/size
            loss_style += 0.5*tf.nn.l2_loss(gram_net-r_styles[layer])
    
    # total loss
            total_loss =  content_weight*loss_content + style_weight*loss_style
    
    return total_loss

# define train
def train(total_loss,learning_rate,global_step):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step=global_step)
    return train_op
    
    