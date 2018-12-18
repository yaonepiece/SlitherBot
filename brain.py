import numpy as np
import tensorflow as tf


class DuelingDQN:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_itr=300,
            batch_size=32,
            e_greedy_increment=None
    ):
        # Parameter Setup
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_itr = replace_target_itr
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.q_target = tf.placeholder(tf.float16, [n_actions], 'Q_Target')
        # Brain Setup
        self.sess = tf.Session()
        self._build_net()

    def _build_net(self):
        # state size is formate the format of NHWC where channel is 4 x frames
        self.state = tf.placeholder(tf.float16, [self.batch_size, 128, 128, 4])
        conv0 = tf.layers.flatten(self.state)
        with tf.name_scope("Conv2D"):
            conv1 = tf.layers.conv2d(input=conv0, filters=16, strides=4, kernel_size=8, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(input=conv1, filters=32, strides=2, kernel_size=4, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(input=conv2, filters=64, strides=1, kernel_size=2, activation=tf.nn.relu)
        with tf.name_scope("Dense"):
            denc0 = tf.layers.dense(conv3, 256)
            denc1 = tf.layers.dense(denc0, 256)
            self.q_eval = tf.layers.dense(denc1, 1)
        with tf.name_scope("Loss"):
            self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_eval)
        with tf.variable_scope('Train'):
            self.train = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def choose_action(self, state):
        return self.sess.run(self.q_eval, {self.state: state})

    def learn(self, action, state, reward):
        pass


def main():
    from gamecontrol import SlitherChromeController
    from screencapture import ScreenCapturer

    print('[DEBUG] Building network')
    brain = DuelingDQN(1)
    data = ScreenCapturer(960, 540, 64, 64)
    game = SlitherChromeController()

    n_round = 1000

    # TODO: get four frame from screen
    print('[DEBUG] Getting initial frames')
    for i in range(4):
        image = data.get_gray()
        data.save_pic(image)

    state = data.screen_data

    # TODO: train the network
    print(f'[DEBUG] Start training... Total round = {n_round}')
    while n_round:
        action = brain.choose_action(state)
        reward = game.perform(action)
        brain.learn(action, state, reward)

        if n_round % 50 == 0:
            print(f'Epoch {n_round} completed, loss={loss_}')

        state[0:2] = state[2:4]
        state[2] = data.get_gray()
        state[3] = data.get_gray()
        n_round -= 1

    # TODO: plot the result (loss graph)
    print('[DEBUG] Train complete')

    # TODO: continue playing
    while True:
        angle = brain.choose_action(state)
        game.turn(angle)


if __name__ == "__main__":
    main()
