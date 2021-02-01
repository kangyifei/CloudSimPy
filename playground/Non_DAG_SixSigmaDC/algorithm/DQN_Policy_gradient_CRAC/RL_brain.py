"""
This part of code is the DQN_Policy_gradient_CRAC brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.compat.v1.set_random_seed(1)

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            scope,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.scope=scope
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.compat.v1.get_collection('target_net_params')
        e_params = tf.compat.v1.get_collection('eval_net_params')
        self.replace_target_op = [tf.compat.v1.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.compat.v1.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        # -------------创建eval神经网络，及时提升参数----------------
        self.s = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s')  # input 用来接收observation
        self.q_target = tf.compat.v1.placeholder(tf.float32, [None, self.n_actions],
                                                 name='Q_target')  # for calculating loss 用来接收q_target的值
        with tf.compat.v1.variable_scope(self.scope+'/eval_net'):
            # c_names(collections_names) are the collections to store variables，在更新target_net参数时会用到
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net，在更新target_net参数时会用到
            # l1 = tf.layers.dense(
            #     inputs=self.s,
            #     units=10,
            #     activation=tf.nn.relu,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='efc1'
            # )
            # l2 = tf.layers.dense(
            #     inputs=l1,
            #     units=20,
            #     activation=tf.nn.relu,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='efc2'
            # )
            # l3 = tf.layers.dense(
            #     inputs=l2,
            #     units=30,
            #     activation=tf.nn.tanh,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='efc3'
            # )
            # l4 = tf.layers.dense(
            #     inputs=l3,
            #     units=20,
            #     activation=tf.nn.tanh,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='efc4'
            # )
            # self.q_eval = tf.layers.dense(
            #     inputs=l4,
            #     units=self.n_actions,
            #     activation=None,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='efc5'
            # )
            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.compat.v1.nn.relu(tf.matmul(self.s, w1) + b1)
            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, 20], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, 20], initializer=b_initializer, collections=c_names)
                l2 = tf.compat.v1.nn.relu(tf.matmul(l1, w2) + b2)
            with tf.compat.v1.variable_scope('l3'):
                w3 = tf.compat.v1.get_variable('w3', [20, 30], initializer=w_initializer, collections=c_names)
                b3 = tf.compat.v1.get_variable('b3', [1, 30], initializer=b_initializer, collections=c_names)
                l3 = tf.compat.v1.nn.relu(tf.matmul(l2, w3) + b3)
            with tf.compat.v1.variable_scope('l4'):
                w4 = tf.compat.v1.get_variable('w4', [30, 20], initializer=w_initializer, collections=c_names)
                b4 = tf.compat.v1.get_variable('b4', [1, 20], initializer=b_initializer, collections=c_names)
                l4 = tf.compat.v1.nn.relu(tf.matmul(l3, w4) + b4)
            # second layer. collections is used later when assign to target net，在更新target_net参数时会用到
            with tf.compat.v1.variable_scope('l5'):
                w5 = tf.compat.v1.get_variable('w5', [20, self.n_actions], initializer=w_initializer,
                                               collections=c_names)
                b5 = tf.compat.v1.get_variable('b5', [1, self.n_actions], initializer=b_initializer,
                                               collections=c_names)
                self.q_eval = tf.matmul(l4, w5) + b5

        with tf.compat.v1.variable_scope('loss'):  # 求误差
            self.loss = tf.reduce_mean(tf.math.squared_difference(self.q_target, self.q_eval))
        with tf.compat.v1.variable_scope('train'):  # 梯度下降
            self._train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net，提供target_Q ------------------
        self.s_ = tf.compat.v1.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.compat.v1.variable_scope(self.scope+'/target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.compat.v1.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            # l1 = tf.layers.dense(
            #     inputs=self.s_,
            #     units=10,
            #     activation=tf.nn.relu,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='tfc1'
            # )
            # l2 = tf.layers.dense(
            #     inputs=l1,
            #     units=20,
            #     activation=tf.nn.relu,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='tfc2'
            # )
            # l3 = tf.layers.dense(
            #     inputs=l2,
            #     units=30,
            #     activation=tf.nn.tanh,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='tfc3'
            # )
            # l4 = tf.layers.dense(
            #     inputs=l3,
            #     units=20,
            #     activation=tf.nn.tanh,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='tfc4'
            # )
            # self.q_next = tf.layers.dense(
            #     inputs=l4,
            #     units=self.n_actions,
            #     activation=None,  # tanh activation
            #     kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            #     bias_initializer=tf.constant_initializer(0.1),
            #     name='tfc5'
            # )
            with tf.compat.v1.variable_scope('l1'):
                w1 = tf.compat.v1.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                               collections=c_names)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            with tf.compat.v1.variable_scope('l2'):
                w2 = tf.compat.v1.get_variable('w2', [n_l1, 20], initializer=w_initializer, collections=c_names)
                b2 = tf.compat.v1.get_variable('b2', [1, 20], initializer=b_initializer, collections=c_names)
                l2 = tf.compat.v1.nn.relu(tf.matmul(l1, w2) + b2)
            with tf.compat.v1.variable_scope('l3'):
                w3 = tf.compat.v1.get_variable('w3', [20, 30], initializer=w_initializer, collections=c_names)
                b3 = tf.compat.v1.get_variable('b3', [1, 30], initializer=b_initializer, collections=c_names)
                l3 = tf.compat.v1.nn.relu(tf.matmul(l2, w3) + b3)
            with tf.compat.v1.variable_scope('l4'):
                w4 = tf.compat.v1.get_variable('w4', [30, 20], initializer=w_initializer, collections=c_names)
                b4 = tf.compat.v1.get_variable('b4', [1, 20], initializer=b_initializer, collections=c_names)
                l4 = tf.compat.v1.nn.relu(tf.matmul(l3, w4) + b4)
            # second layer. collections is used later when assign to target net
            with tf.compat.v1.variable_scope('l5'):
                w5 = tf.compat.v1.get_variable('w5', [20, self.n_actions], initializer=w_initializer,
                                               collections=c_names)
                b5 = tf.compat.v1.get_variable('b5', [1, self.n_actions], initializer=b_initializer,
                                               collections=c_names)
                self.q_next = tf.matmul(l4, w5) + b5

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs,dtype=np.float)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



