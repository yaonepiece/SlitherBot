import tensorflow as tf
import numpy as np
import cv2
import os


def image_read(file):
    n_array = cv2.imread(file, 1)

    if n_array is None:
        return None

    shape_max = max(n_array.shape[0], n_array.shape[1])

    n_array = np.pad(n_array, (
        ((shape_max-n_array.shape[0])>>1, (shape_max-n_array.shape[0])-((shape_max-n_array.shape[0])>>1)),
        ((shape_max-n_array.shape[1])>>1, (shape_max-n_array.shape[1])-((shape_max-n_array.shape[1])>>1)),
        (0,0)
        ), 'constant', constant_values=(0))

    res = cv2.resize(n_array, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

    return res


def network(x):
    linear = tf.layers.flatten(x)
    with tf.name_scope("Encoder"):
        enc1 = tf.layers.dense(linear, 1024*3)
        enc2 = tf.layers.dense(enc1, 256*3)
        enc3 = tf.layers.dense(enc2, 64*3)
    with tf.name_scope("Decoder"):
        dec1 = tf.layers.dense(enc3, 256*3)
        dec2 = tf.layers.dense(dec1, 1024*3)
        dec3 = tf.layers.dense(dec2, 4096*3)
    loss = tf.losses.mean_squared_error(labels=linear, predictions=dec3)
    train = tf.train.AdamOptimizer(0.01).minimize(loss)
    return dec3, loss, train


img_dir = '/Users/mac/Desktop'

ls = os.listdir(img_dir)
ld_img = []

for file in ls:
    if file[-4:] == '.jpg':
        img = image_read(img_dir+'/'+file)
        if img is not None:
            ld_img += [img]

print('The file input is completed.')

ld_img = np.asarray(ld_img)

round=100

with tf.Session() as sess:
    print('Start building network')
    x = tf.placeholder(tf.float64, shape=(None, 64, 64, 3))
    _, loss, train = network(x)
    print('Network built.')
    sess.run(tf.initializers.global_variables())
    loss_ = sess.run(loss, {x: ld_img})
    print('Initial loss:')
    print(loss_)
    print('Start training... total round = {0}'.format(round))
    for epoch in range(round):
        sess.run(train, {x: ld_img})
        if epoch%1000 == 999:
            print('Round {0} completed.'.format(epoch+1))
    print('Train complete')
    loss_ = sess.run(loss, {x: ld_img})
    print('Loss:')
    print(loss_)
