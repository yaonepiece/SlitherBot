import screencapture as sc

import tensorflow as tf
import numpy as np
from PIL import ImageGrab
import cv2, os, time, select

scap=sc.ScreenCapturer(960,540,64,64)
round = 1000

#TODO: change some dense layers into CNN
#      put the function within a class called Brain,
#      and complete the whole class

def network(x):
    linear = tf.layers.flatten(x)
    with tf.name_scope("Eyes"):
        enc1 = tf.layers.dense(linear, 16*16*4)
        enc2 = tf.layers.dense(enc1, 8*8*4)
        enc3 = tf.layers.dense(enc2, 4*4*4)
    with tf.name_scope("Brain"):
        dec1 = tf.layers.dense(enc3, 64*64*4)
    loss = tf.losses.mean_squared_error(labels=linear, predictions=dec1)
    train = tf.train.AdamOptimizer(0.0001).minimize(loss)
    return enc3, dec1, loss, train

with tf.Session() as sess:
    for i in range(4):
        screen=scap.get_gray()
        scap.save_pic(screen)

    scr_tf = [np.transpose(scap.screen_data, (1,2,0))]
    print('Start building network')
    x = tf.placeholder(tf.float64, shape=(None, 64, 64, 4))
    encoder, decoder, loss, train = network(x)
    img_tensor=tf.reshape(decoder,[-1,4,4,3])
    print('Network built.')
    sess.run(tf.initializers.global_variables())
    loss_ = sess.run(loss, {x: scr_tf})
    print('Initial loss: ',end='')
    print(loss_)
    print('Start training... total round = {0}'.format(round))

    for epoch in range(round):
        sess.run(train, {x:scr_tf})
        loss_=sess.run(loss, {x: scr_tf})
        if (epoch+1)%50 == 0:
            print('Epoch {0} completed, loss='.format(epoch+1),end='')
            print(loss_)
            
    print('Train complete')
    loss_ = sess.run(loss, {x: scr_tf})
    print('Loss:')
    print(loss_)
