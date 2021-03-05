import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random

BATCH_SIZE = 32
LR = 5e-4
START_EPSILON = 0.1
DELAY_EPSILON = 0.999
FINAL_EPSILON = 0.01
EPSILON = START_EPSILON
GAMMA = 0.99
MEMORY_SIZE = int(1e5)
MEMORY_THRESHOLD = int(1e3)
SYNE_FREQUENCY = int(500)
LEARN_FREQUENCY = int(100)
import numpy


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 1e-6
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 1e-6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_network = DuelingNet(num_feature, num_actions).to(self.device)
        self.target_network = DuelingNet(num_feature, num_actions).to(self.device)
        self.num_actions = num_actions
        self.memory = Memory(MEMORY_SIZE)
        self.optimizer = torch.optim.Adam(self.eval_network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.times = 0

    def action(self, state, israndom):
        global EPSILON
        if israndom and random.random() < EPSILON:
            if EPSILON > FINAL_EPSILON:
                EPSILON = EPSILON * DELAY_EPSILON
            return np.random.randint(0, state.shape[0])
        state = torch.FloatTensor(state).to(self.device)
        actions_value = self.eval_network.forward(state).cpu()
        if EPSILON > FINAL_EPSILON:
            EPSILON = EPSILON * DELAY_EPSILON
        return torch.max(actions_value, 0)[1].item()

    @torch.no_grad()
    def target_net_eval(self, state):
        return self.target_network(state)

    @torch.no_grad()
    def append_sample(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        eval_q = self.eval_network.forward(state_t.view(1, -1))
        eval_q_actul = eval_q.item()
        target_q = self.target_net_eval(next_state_t.view(1, -1))
        target_q = target_q.view(-1).item()
        target_q_actul = reward + GAMMA * target_q * done
        error = abs(eval_q_actul - target_q_actul)
        self.memory.add(error, (state, action, reward, next_state, done))

    def learn(self, state, action, reward, next_state, done):
        self.times += 1
        if done:
            self.append_sample(state, action, reward, next_state, 0)
        else:
            self.append_sample(state, action, reward, next_state, 1)
        if self.times < MEMORY_THRESHOLD:
            # print("mem not full ")
            return
        if self.times % SYNE_FREQUENCY == 0:
            self.target_network.load_state_dict(self.eval_network.state_dict())
        if self.times % LEARN_FREQUENCY == 0:
            batch, idx, is_weight = self.memory.sample(BATCH_SIZE)

            is_weight = torch.FloatTensor(is_weight).to(self.device)
            state = torch.FloatTensor([x[0] for x in batch]).to(self.device)
            action = torch.LongTensor([[x[1]] for x in batch]).to(self.device)
            reward = torch.FloatTensor([[x[2]] for x in batch]).to(self.device)
            next_state = torch.FloatTensor([x[3] for x in batch]).to(self.device)
            done = torch.FloatTensor([[x[4]] for x in batch]).to(self.device)

            # 计算eval网络中每个动作对应的Q值
            eval_q = self.eval_network.forward(state)
            # 只留下实际发生动作的q值
            # eval_q_actul = eval_q.gather(1, action)
            # 计算eval网络中应该选择的动作
            # action_from_eval_q = eval_q.max(1)[1].view(-1, 1)
            target_q = self.target_net_eval(next_state)
            # target_q = target_q.gather(1, action_from_eval_q)
            target_q_actul = reward + GAMMA * target_q.view(BATCH_SIZE, 1) * done
            weights = abs(target_q_actul - eval_q)
            for i in range(BATCH_SIZE):
                self.memory.update(idx[i], weights[i].item())
            loss = (self.loss_func(eval_q, target_q_actul) * is_weight).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
