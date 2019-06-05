import random
import numpy as np
import tensorflow as tf
from screenCapture import ScreenCapturer
import gameControl

class DQNBrain:
    def __init__(
            self,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_itr=100,
            batch_size=32,
            e_greedy_increment=None
    ):
        # Parameter Setup
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        # self.replace_target_itr = replace_target_itr # update target network only when the game is lost to prevent lagging
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 1 if e_greedy_increment is None else e_greedy
        
        # Brain Setup
        self.sess = tf.Session()
        self.q_real = tf.placeholder(tf.float16, [None, n_actions], 'Q_Real')
        self._build_net()
        # self.state: game screen input, shape=[self.batch_size, 128, 128, 4]
        # self.target_net: network for action, feed_dict={self.state: self.state_mem}
        # self.eval_net: network for training, feed_dict={self.state: self.state_mem}
        # self.loss: loss function for eval_net, feed_dict={self.state: self.state_mem, self.q_real: self.q_mem}
        # self.train: loss optimizer, feed_dict={self.state: self.state_mem, self.q_real: self.q_mem}
        # self.replace_op: update the target network with evaluation network
        self.sess.run(tf.initializers.global_variables())
        
        # Action Memory
        self.mem_size=0
        self.state_mem=[]
        #self.reward_mem=[]
        self.last_reward=None
        self.last_action=None
        self.q_mem=[]

    def _build_net(self):
        # state size is formatted in NHWC
        self.state = tf.placeholder(tf.float16, [None, 128, 128, 4])
        with tf.variable_scope('eval_net'):
            with tf.name_scope('Conv2D'):
                conv0 = tf.layers.conv2d(self.state, filters=16, strides=4, kernel_size=8, activation=tf.nn.relu)
                conv1 = tf.layers.conv2d(conv0, filters=32, strides=2, kernel_size=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(conv1, filters=64, strides=1, kernel_size=2, activation=tf.nn.relu)
                conv3 = tf.layers.flatten(conv2)
            with tf.name_scope('Dense'):
                denc0 = tf.layers.dense(conv3, 256)
                denc1 = tf.layers.dense(denc0, 256)
                self.q_eval = tf.layers.dense(denc1, self.n_actions)
        with tf.variable_scope('target_net'):
            with tf.name_scope('Conv2D'):
                conv0 = tf.layers.conv2d(self.state, filters=16, strides=4, kernel_size=8, activation=tf.nn.relu)
                conv1 = tf.layers.conv2d(conv0, filters=32, strides=2, kernel_size=4, activation=tf.nn.relu)
                conv2 = tf.layers.conv2d(conv1, filters=64, strides=1, kernel_size=2, activation=tf.nn.relu)
                conv3 = tf.layers.flatten(conv2)
            with tf.name_scope('Dense'):
                denc0 = tf.layers.dense(conv3, 256)
                denc1 = tf.layers.dense(denc0, 256)
                self.q_target = tf.layers.dense(denc1, self.n_actions)
        with tf.name_scope('Loss'):
            self.loss = tf.losses.mean_squared_error(labels=self.q_real, predictions=self.q_eval)
        with tf.name_scope('Train'):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        with tf.name_scope('Replace'):
            self.replace_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]

    def choose_action(self, state):
        if self.epsilon>=1 or random.random()<self.epsilon:
            values=self.sess.run(self.q_target, {self.state: [state]}).ravel()
            return np.argmax(values)
        else:
            return random.randrange(self.n_actions)

    def learn(self, action, before_state, after_state, reward):
        prediction=self.sess.run(self.q_target, {self.state: [before_state]}).ravel()
        if after_state is not None:
            pred_next=self.sess.run(self.q_target, {self.state: [after_state]}).ravel()
            # Q-Learning
            # prediction[action]=self.gamma*np.max(pred_next)+reward
            # Sarsa
            # prediction[action]=self.gamma*np.random.choice(pred_next)+reward
            # Assume the nature action is to continue the previous action
            prediction[action]=self.gamma*pred_next[action]+reward
        else:
            prediction[action]=reward
        self.state_mem.append(before_state)
        self.q_mem.append(prediction)
        self.mem_size+=1
        if self.mem_size>self.batch_size:
            self.state_mem=self.state_mem[-self.batch_size:]
            self.q_mem=self.q_mem[-self.batch_size:]
            self.mem_size=self.batch_size
        if self.mem_size==self.batch_size:
            self.sess.run(self.train, {self.state: self.state_mem, self.q_real: self.q_mem})
            if self.epsilon < 1 and self.epsilon_increment is not None:
                self.epsilon+=self.epsilon_increment
    
    def update_network(self):
        self.sess.run(self.replace_op)

def play(brain, data):
    n_round = 1000
    update_round = 1
    log_round = 10
    
    for i in range(1,n_round+1):
        while gameControl.status==0:
            pass
        
        # get four frame from screen
        print(' [LOG]  Training started, getting initial frames...')
        while not data.data_ready:
            data.save_pic(data.get_gray())
        before_state = np.transpose(data.screen_data,axes=(1,2,0))
        before_score = gameControl.score
        
        while True:
            action = brain.choose_action(before_state)
            gameControl.action = action
            # time.sleep(0.1)
            
            data.save_pic(data.get_gray())
            after_state = np.transpose(data.screen_data,axes=(1,2,0))
            after_score = gameControl.score
            if not gameControl.status==0:
                brain.learn(action, before_state, after_state, (after_score-before_score)/10)
            else:
                brain.learn(action, before_state, None, -before_score/10)
                break
            
            before_state = after_state
            before_score = after_score
        
        if i % update_round == 0:
            print(' [LOG]  Updating the network...')
            brain.update_network()
            print(' [LOG]  Update finished.')
        if i % log_round == 0:
            print(f' [LOG]  Round {i} completed, score = {before_score}')
        
        data.clear()
        
    # TODO: plot the result (loss graph)
    print(' [LOG]  Train complete')

def main():
    import threading
    import time

    print(' [LOG]  Creating environment...')
    brain = DQNBrain(4)
    print(' [LOG]  Network created.')
    # modify the crop rect to the center of the game screen
    # first 2 numbers are the center pixel of the image
    # following 2 numbers are the size of the rect to be cropped
    # last 2 numbers are the actual image output size, should be left as (128,128)
    data = ScreenCapturer(480, 540, 256, 256, outx=128, outy=128)
    print(' [LOG]  Screen capturer created.')
    player = threading.Thread(target=play,args=(brain, data))
    player.start()
    print(' [LOG]  Server created, waiting for the slither.')
    gameControl.main()

if __name__ == "__main__":
    main()
