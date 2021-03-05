from core.algorithm import Algorithm
from core.config import MachineConfig
from core.machine import Machine
from core.cluster import Cluster
from typing import List
from .RL_brain import DeepQNetwork
import numpy as np


class DQNAlgorithm(Algorithm):
    def __init__(self, machine_configs: List[MachineConfig]):
        self.machine_configs = machine_configs
        self.dqn = DeepQNetwork(n_actions=len(machine_configs),
                                n_features=len(machine_configs) * 2 + 2,
                                learning_rate=0.01, e_greedy=0.9,
                                replace_target_iter=100, memory_size=20000,
                                e_greedy_increment=0.001, )
        self.steps = 0
        self.last_observation = []
        for machine_config in self.machine_configs:
            self.last_observation.append(machine_config.cpu)
            self.last_observation.append(machine_config.memory)
        self.last_observation.append(0)
        self.last_observation.append(0)
        self.last_action = 0
        self.last_reward = 0

        self.last_clock = 0
        self.last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0

    def __call__(self, cluster, clock, cooling_equip=None):
        self.total_called_num += 1
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        if not len(tasks) == 0:
            observation = []
            for machine in machines:
                observation.append(machine.cpu)
                observation.append(machine.memory)
            observation.append(tasks[0].task_config.cpu)
            observation.append(tasks[0].task_config.memory)
            observation = np.array(observation)
            action = self.dqn.choose_action(observation)
            if not machines[action].accommodate(tasks[0]):
                reward = -100
                self.total_reward_of_ep += reward
                self.dqn.store_transition(self.last_observation, self.last_action, self.last_reward, observation)
                self.last_observation = observation
                self.last_action = action
                self.last_reward = reward
                self.steps += 1
                return None, None, None
            else:
                reward = self.reward_giver(cluster, clock)
                self.total_reward_of_ep += reward
                self.dqn.store_transition(self.last_observation, self.last_action, self.last_reward, observation)
                self.last_observation = observation
                self.last_action = action
                self.last_reward = reward
                self.steps += 1
                return machines[action], tasks[0], None
        return None, None, None

    def cal_machine_power(self, machine: Machine):
        cpu_usage = machine.state["cpu_usage_percent"]
        memory_usage = machine.state['memory_usage_percent']
        return 100 * cpu_usage ** 1.9 + 5 * memory_usage

    def reward_giver(self, cluster: Cluster, clock):
        # newpower = 0
        # for machine in cluster.machines:
        #     newpower += self.cal_machine_power(machine)
        # energy_consume = self.last_power * (clock - self.last_clock)
        # self.last_clock = clock
        # self.last_power = newpower
        # self.total_energy_consume += energy_consume
        energy_consume = cluster.monitor.total_energy_consume
        self.total_energy_consume += energy_consume
        if energy_consume == 0:
            return 0
        else:
            return -energy_consume

    def reset(self):
        self.last_observation = []
        for machine_config in self.machine_configs:
            self.last_observation.append(machine_config.cpu)
            self.last_observation.append(machine_config.memory)
        self.last_observation.append(0)
        self.last_observation.append(0)
        self.last_clock = 0
        self.last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
