#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:51:20 2017

@author: li
"""
import tensorflow as tf
import numpy as np
import scipy.misc
import neural_style
import os
from datetime import datetime
import time
import vgg
from argparse import ArgumentParser

# define global variables
ITERATIONS = 50
CONTENT_WEIGHT = 1e-3
STYLE_WEIGHT = 1e-2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0


# define parser
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content',dest='content',
                        help='path to content image',
                        metavar='CONTENT',required=True)
    parser.add_argument('--style',dest='style',
                        help='path to style image',
                        metavar='STYLE',required=True)
    parser.add_argument('--output',dest='output',
                        help='path to output image',
                        metavar='OUTPUT',required=True)
    parser.add_argument('--network',dest='network',
                        help='path to pretrained network',
                        metavar='NETWORK',required=True)
    parser.add_argument('--iterations', type=int, dest='iterations', 
                        help='iterations (default %(default)s)',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--trian-dir', dest='train_dir', 
                        help='dir for checkpoint', metavar='CHECKPOINT', 
                        default='/tmp/neural_style')
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('--style-scales', type=float,dest='style_scales',
                        help='one or more style scales',
                        metavar='STYLE_SCALE',default=STYLE_SCALE)
    parser.add_argument('--content-weight', type=float,dest='content_weight', 
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float, dest='style_weight', 
                        help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', 
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)
    return parser

#define train
def train():
    parser = build_parser()
    options = parser.parse_args()
    
    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)
    
    # read the content and style image
    content_image = imread(options.content)
    style_image = imread(options.style)
    # resize th style_image with style_scale
    style_image = scipy.misc.imresize(style_image,content_image.shape[1]*
                                      options.style_scales/style_image.shape[1])
    # pass through inference
    r_content,r_styles,net,mean_pixel = neural_style.inference(content_image,
                                        style_image,options.network)
    # get loss
    total_loss = neural_style.loss(r_content,r_styles,net,options.content_weight,
                                   options.style_weight)
    # get train op
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = neural_style.train(total_loss,options.learning_rate,global_step)
    # define a hook for log and checkpoint
    class _LoggerHook(tf.train.SessionRunHook):
        
        def begin(self):
            self._step = -1
        
        def before_run(self, run_context):
            self._step += 1
            self._start_time = time.time()
            if self._step % 40 == 0:
                image = tf.get_default_graph().get_tensor_by_name('image:0')
                return tf.train.SessionRunArgs({'t_loss':total_loss,'image': image})  # Asks for loss value.
        
        def after_run(self, run_context, run_values):
            
            if self._step % 40 == 0:
        
                loss_value = run_values.results['t_loss']
                image2 = run_values.results['image']
                image2 = vgg.unprocess(image2.reshape(image2.shape[1:]), mean_pixel)
                imsave(options.output,image2)
                duration = time.time() - self._start_time
                format_str = ('%s: step %d, loss = %.2f , %.3f sec/step')
                print (format_str % (datetime.now(), self._step, loss_value,
                                     duration))

    with tf.device('/cpu:0'),tf.train.MonitoredTrainingSession(
            checkpoint_dir=options.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=options.iterations),
            tf.train.NanTensorHook(total_loss),_LoggerHook()]) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(train_op)


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

if __name__ == '__main__':
    train()