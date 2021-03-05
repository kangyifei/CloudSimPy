from core.algorithm import Algorithm
from core.config import MachineConfig
from core.machine import Machine
from core.cluster import Cluster
from typing import List
from .RL_brain import Agent
import numpy as np


class DQNAlgorithm(Algorithm):
    def __init__(self, machine_configs: List[MachineConfig]):
        self.machine_configs = machine_configs
        self.dqn = Agent(3 + 4, 1)
        self.steps = 0
        self.last_observation = []
        for _ in range(7):
            self.last_observation.append(0)
        self.last_action = 0
        self.last_reward = 0
        with open("./jobs_info.csv") as f:
            self.jobs_info=f.readline()
        self.jobs_info=self.jobs_info.split(",")
        self.jobs_info = [float(i) for i in self.jobs_info]
        self.jobs_info[4]=1 if self.jobs_info[4]==0 else self.jobs_info[4]
        self.jobs_info[6] = 1 if self.jobs_info[6] == 0 else self.jobs_info[6]
        self.jobs_info[8] = 1 if self.jobs_info[8] == 0 else self.jobs_info[8]
        self.jobs_info[10] = 1 if self.jobs_info[10] == 0 else self.jobs_info[10]
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
                        observation.append((task.task_config.cpu - self.jobs_info[5]) / (self.jobs_info[6]**0.5))
                        observation.append((task.task_config.memory - self.jobs_info[7]) / (self.jobs_info[8]**0.5))
                        observation.append((task.task_config.duration - self.jobs_info[9]) / (self.jobs_info[10]**0.5))
                        observation.append((task.task_config.instances_number - self.jobs_info[3]) / (self.jobs_info[4]**0.5))
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
        if energy_consume<0:
            print("under 0")
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
        self.last_last_power=0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
