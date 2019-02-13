import random
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
        self.replace_target_itr = replace_target_itr
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 1 if e_greedy_increment is None else e_greedy
        self.q_target = tf.placeholder(tf.float16, [n_actions], 'Q_Target')
        
        # Brain Setup
        self.sess = tf.Session()
        self._build_net()
        self.sess.run(tf.initializers.global_variables())
        
        # Action Memory
        self.mem_size=0
        self.state_mem=[]
        #self.reward_mem=[]
        self.last_reward=None
        self.last_action=None
        self.q_mem=[]

    def _build_net(self):
        # state size is formate the format of NHWC where channel is 4 x frames
        # Note: no deep(double) structure currently
        # 2/10: Can't find a structure of Dueling Q Net
        self.state = tf.placeholder(tf.float16, [self.batch_size, 128, 128, 4])
        conv0 = tf.layers.flatten(self.state)
        with tf.name_scope("Conv2D"):
            conv1 = tf.layers.conv2d(input=conv0, filters=16, strides=4, kernel_size=8, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(input=conv1, filters=32, strides=2, kernel_size=4, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(input=conv2, filters=64, strides=1, kernel_size=2, activation=tf.nn.relu)
        with tf.name_scope("Dense"):
            denc0 = tf.layers.dense(conv3, 256)
            denc1 = tf.layers.dense(denc0, 256)
            self.q_eval = tf.layers.dense(denc1, self.n_actions)
        with tf.name_scope("Loss"):
            self.loss = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q_eval)
        with tf.name_scope('Train'):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, state):
        if random.random()<self.epsilon:
            values=self.sess.run(self.q_eval, {self.state: state}).ravel()
            return np.amax(values)
        else:
            return random.randrange(self.n_actions)

    def learn(self, action, before_state, after_state, reward):
        prediction=self.sess.run(self.q_eval, {self.state: before_state}).ravel()
        pred_next=self.sess.run(self.q_eval, {self.state: after_state}).ravel()
        prediction[action]=self.gamma*np.max(pred_next)+reward
        self.state_mem.append(before_state)
        self.q_mem.append(prediction)
        self.mem_size+=1
        if self.mem_size>self.batch_size:
            self.state_mem=[-self.batch_size:]
            self.q_mem=[-self.batch_size:]
            self.mem_size=self.batch_size
        if self.mem_size==self.batch_size:
            self.sess.run(self.train, {self.state: self.state_mem, self.q_target: self.q_mem})
            if self.epsilon < 1 and self.epsilon_increment is not None:
                self.epsilon+=self.epsilon_increment


def main():
    from gamecontrol import SlitherChromeController
    from screencapture import ScreenCapturer

    print('[DEBUG] Building network')
    brain = DuelingDQN(1)
    data = ScreenCapturer(960, 540, 64, 64)
    game = SlitherChromeController('localhost', 8765)

    n_round = 1000

    # TODO: get four frame from screen
    print('[DEBUG] Getting initial frames')
    while not data.data_ready:
        image = data.get_gray()
        data.save_pic(image)

    state = data.screen_data

    # TODO: train the network
    print(f'[DEBUG] Start training... Total round = {n_round}')
    while n_round:
        action = brain.choose_action(state)
        reward = game.turn(action)
        brain.learn(action, state, reward)

        if n_round % 50 == 0:
            print(f'Epoch {n_round} completed, loss={loss_}')

        frame = data.get_gray()
        data.save_pic(frame)

        n_round -= 1

    # TODO: plot the result (loss graph)
    print('[DEBUG] Train complete')

    # TODO: continue playing
    while True:
        angle = brain.choose_action(state)
        game.turn(angle)


if __name__ == "__main__":
    main()