import screencapture as sc
import tensorflow as tf
import numpy as np
from PIL import ImageGrab
import cv2, os, time, select


class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_itr=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_itr = replace_target_itr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.state = 1
        self._build_net()

    def _build_net(self):
        conv0 = tf.layers.flatten(self.state)
        with tf.name_scope("Conv2D"):
            conv1 = tf.layers.conv2d(input=conv0, filters=16, strides=4, kernel_size=8, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(input=conv1, filters=32, strides=2, kernel_size=4, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(input=conv2, filters=64, strides=1, kernel_size=2, activation=tf.nn.relu)
        with tf.name_scope("Dense"):
            denc0 = tf.layers.dense(conv3, 128)
            denc1 = tf.layers.dense(denc0, 128)
            denc2 = tf.layers.dense(denc1, 1)
        with tf.name_scope("Loss"):
            loss = tf.losses.mean_squared_error(labels=linear, predictions=denc2)
            train = tf.train.AdamOptimizer(0.0001).minimize(loss)
        return loss, train

    def choose_action(self):
        pass


def main():
    scap = sc.ScreenCapturer(960, 540, 64, 64)
    round = 1000

    with tf.Session() as sess:
        for i in range(4):
            screen = scap.get_gray()
            scap.save_pic(screen)

        scr_tf = [np.transpose(scap.screen_data, (1, 2, 0))]
        print('Start building network')
        x = tf.placeholder(tf.float64, shape=(None, 64, 64, 4))
        encoder, decoder, loss, train = network(x)
        img_tensor = tf.reshape(decoder, [-1, 4, 4, 3])
        print('Network built.')
        sess.run(tf.initializers.global_variables())
        loss_ = sess.run(loss, {x: scr_tf})
        print('Initial loss: ', end='')
        print(loss_)
        print('Start training... total round = {0}'.format(round))

        for epoch in range(round):
            sess.run(train, {x: scr_tf})
            loss_ = sess.run(loss, {x: scr_tf})
            if (epoch + 1) % 50 == 0:
                print('Epoch {0} completed, loss='.format(epoch + 1), end='')
                print(loss_)

        print('Train complete')
        loss_ = sess.run(loss, {x: scr_tf})
        print('Loss:')
        print(loss_)


if __name__ == "__main__":
    main()
