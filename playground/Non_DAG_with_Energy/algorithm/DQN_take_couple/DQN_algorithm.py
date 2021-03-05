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
        self.dqn = DeepQNetwork(n_actions=1,
                                n_features=3+4,
                                learning_rate=0.01, e_greedy=0.9,
                                replace_target_iter=100, memory_size=20000,
                                e_greedy_increment=0.001, batch_size=128)
        self.steps = 0
        self.last_observation = []
        for _ in range(7):
            self.last_observation.append(0)
        self.last_action = 0
        self.last_reward = 0

        self.last_clock = 1
        self.last_power = 0
        self.last_last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0

    def __call__(self, cluster, clock, cooling_equip=None):
        self.total_called_num += 1
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        observation_list = []
        machine_task_pair = []
        if not len(tasks) == 0:
            for task in tasks:
                for machine in machines:
                    if machine.accommodate(task):
                        observation = []
                        machine_task_pair.append((machine, task))
                        state = machine.state
                        observation.append((state["cpu_usage_percent"] - 0.5) / 0.08333333333333333)
                        observation.append((state["memory_usage_percent"] - 0.5) / 0.08333333333333333)
                        observation.append((state["power"] - 200) / 833.3333333333334)
                        observation.append((task.task_config.cpu - 0.5264810426540284) / 0.316514460288987)
                        observation.append((task.task_config.memory - 0.009175121384696406) / 0.05250922914043805)
                        observation.append((task.task_config.duration - 74.68815165876777) / 6.738206767568669)
                        observation.append((task.task_config.instances_number - 45.376344086021504) / 9.245963255457019)
                        observation_list.append(observation)
            if len(observation_list) == 0:
                return None, None, None
            action = self.dqn.action(np.array(observation_list), True)
            machine, task = machine_task_pair[action]
            done = False
            if len(tasks) == 1:
                if task.waiting_task_instances_number == 1:
                    done = True
            reward = self.reward_giver(cluster, clock)
            self.total_reward_of_ep += reward
            self.dqn.learn(self.last_observation, self.last_action, reward, observation_list[action], done)
            self.last_observation = observation_list[action]
            self.last_action = action
            self.last_reward = reward
            return machine, task, None
        return None, None, None

    def reward_giver(self, cluster: Cluster, clock):
        newpower = 0
        for machine in cluster.machines:
            newpower += machine.state["power"]
        if newpower == self.last_power:
            # 选择不分配
            return 0
        if clock == self.last_clock:
            # 计算同一时间步能耗（功率）的增加
            energy_consume = self.last_power - self.last_last_power
        else:
            # 计算上一个动作带来的能耗
            energy_consume = self.last_power * (clock - self.last_clock)
            self.total_energy_consume += energy_consume
        # energy_consume = self.last_power - self.last_last_power
        self.last_clock = clock
        self.last_power = newpower
        self.last_last_power = self.last_power
        # power_sum=0
        # machine_state_list = cluster.state['machine_states']
        # for machine_state in machine_state_list:
        #     power_sum += machine_state["power"]
        # energy_consume=cluster.monitor.total_energy_consume-self.total_energy_consume
        # self.total_energy_consume += energy_consume
        if energy_consume == 0:
            return 0
        else:
            return 250 / energy_consume

    def reset(self):
        self.last_observation = []
        for _ in range(7):
            self.last_observation.append(0)
        self.last_clock = 1
        self.last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
