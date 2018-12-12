import screencapture as sc

import tensorflow as tf
import numpy as np
from PIL import ImageGrab
import cv2, os, time, select


def network(x):
	linear = tf.layers.flatten(x)
	with tf.name_scope("Encoder"):
		enc1 = tf.layers.dense(linear, 16*16*3)
		enc2 = tf.layers.dense(enc1, 8*8*3)
		enc3 = tf.layers.dense(enc2, 4*4*3)
	with tf.name_scope("Decoder"):
		dec1 = tf.layers.dense(enc3, 64*64*3)
	loss = tf.losses.mean_squared_error(labels=linear, predictions=dec1)
	train = tf.train.AdamOptimizer(0.0001).minimize(loss)
	return enc3, dec1, loss, train


scap=sc.ScreenCapturer(960,540,64,64)
round = 1000

with tf.Session() as sess:
	screen=scap.get_pic()
	scr_tf = np.reshape(screen, (1, 64,64,3))
	print('Start building network')
	x = tf.placeholder(tf.float64, shape=(1, 64, 64, 3))
	encoder, decoder, loss, train = linear(x)
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
	
