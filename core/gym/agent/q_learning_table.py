import numpy as np
import pandas as pd


class QLearningTable:
    #初始化
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #初始化，空的dataframe

    #选行为
    def choose_action(self, observation):
        self.check_state_exist(observation) #检测本state是否在q_table中存在
        # action selection
        if np.random.uniform() < self.epsilon:  #选择Q value中最高的action
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            # 同一个 state, 可能会有多个相同的 Q action value, 所以我们乱序一下
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    #学习更新参数
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)  #检测q_table中是否存在s_
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新对应的 state-action 值
        #print(self.q_table)

    #检测state是否存在
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            #如果没有当前的state，就插入一个全0数据，当作这个state的所有action初始values
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )