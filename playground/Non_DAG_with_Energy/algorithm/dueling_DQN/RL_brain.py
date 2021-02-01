import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random

BATCH_SIZE = 32
LR = 0.001
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
EPSILON = START_EPSILON
EXPLORE = 1000000
GAMMA = 0.99
TOTAL_EPISODES = 10000000
MEMORY_SIZE = 1000000
MEMORY_THRESHOLD = 100000
TEST_FREQUENCY = 1000


class DuelingNet(nn.Module):

    def __init__(self, num_features, num_actions):
        super(DuelingNet, self).__init__()
        self.num_actions = num_actions
        self.l1 = nn.Sequential(
            nn.Linear(num_features, 20),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU()
        )
        self.l3 = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU()
        )
        self.hidden_adv = nn.Sequential(
            nn.Linear(100, 512, bias=True),
            nn.ReLU()
        )
        self.hidden_val = nn.Sequential(
            nn.Linear(100, 512, bias=True),
            nn.ReLU()
        )
        self.adv = nn.Sequential(
            nn.Linear(512, num_actions, bias=True)
        )
        self.val = nn.Sequential(
            nn.Linear(512, 1, bias=True)
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.0, 0.02)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        adv = self.hidden_adv(x)
        val = self.hidden_val(x)

        adv = self.adv(adv)
        val = self.val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x


class Agent(object):
    def __init__(self, num_feature, num_actions):
        self.network = DuelingNet(num_feature, num_actions)
        self.num_actions=num_actions
        self.memory = deque()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def action(self, state, israndom):
        if israndom and random.random() < EPSILON:
            return np.random.randint(0, self.num_actions)
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        actions_value = self.network.forward(state)
        return torch.max(actions_value, 1)[1].data.numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        if done:
            self.memory.append((state, action, reward, next_state, 0))
        else:
            self.memory.append((state, action, reward, next_state, 1))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()
        if len(self.memory) < MEMORY_THRESHOLD:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        state = torch.FloatTensor([x[0] for x in batch])
        action = torch.LongTensor([[x[1]] for x in batch])
        reward = torch.FloatTensor([[x[2]] for x in batch])
        next_state = torch.FloatTensor([x[3] for x in batch])
        done = torch.FloatTensor([[x[4]] for x in batch])

        eval_q = self.network.forward(state).gather(1, action)
        next_q = self.network(next_state).detach()
        target_q = reward + GAMMA * next_q.max(1)[0].view(BATCH_SIZE, 1) * done
        loss = self.loss_func(eval_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()