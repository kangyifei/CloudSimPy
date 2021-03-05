from core.algorithm import Algorithm
from core.config import MachineConfig
from core.machine import Machine
from core.cluster import Cluster
from typing import List
from .RL_brain import Agent
import numpy as np
import json


class CRACDQNAlgorithm():
    def __init__(self, machine_configs: List[MachineConfig]):
        self.machine_configs = machine_configs
        self.dqn = Agent(len(self.machine_configs) * 3 + 4, 100)
        self.steps = 0
        self.last_observation = []
        for _ in range(7):
            self.last_observation.append(0)
        self.last_action = 0
        self.last_reward = 0
        with open("./jobs_info.csv") as f:
            self.jobs_info = f.readline()
        self.jobs_info = self.jobs_info.split(",")
        self.jobs_info = [float(i) for i in self.jobs_info]
        self.jobs_info[4] = 1 if self.jobs_info[4] == 0 else self.jobs_info[4]
        self.jobs_info[6] = 1 if self.jobs_info[6] == 0 else self.jobs_info[6]
        self.jobs_info[8] = 1 if self.jobs_info[8] == 0 else self.jobs_info[8]
        self.jobs_info[10] = 1 if self.jobs_info[10] == 0 else self.jobs_info[10]
        with open("./tetris.json")as f:
            self.baseline = json.load(f)
        self.last_clock = 0
        self.last_power = 0
        self.last_last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0

    def __call__(self, cluster, task, done, clock, cooling_equip):
        machines = cluster.machines
        self.total_called_num += 1
        observation = []
        for machine in machines:
            state = machine.state
            observation.append((state["cpu_usage_percent"] - 0.5) / 0.08333333333333333)
            observation.append((state["memory_usage_percent"] - 0.5) / 0.08333333333333333)
            observation.append((state["power"] - 200) / 833.3333333333334)
        observation.append((task.task_config.cpu - self.jobs_info[5]) / (self.jobs_info[6] ** 0.5))
        observation.append((task.task_config.memory - self.jobs_info[7]) / (self.jobs_info[8] ** 0.5))
        observation.append(
            (task.task_config.duration - self.jobs_info[9]) / (self.jobs_info[10] ** 0.5))
        observation.append(
            (task.task_config.instances_number - self.jobs_info[3]) / (self.jobs_info[4] ** 0.5))
        action = self.dqn.action(np.array(observation), True)
        reward = self.reward_giver(cooling_equip, clock)
        self.total_reward_of_ep += reward
        self.dqn.learn(self.last_observation, self.last_action, reward, observation, done)
        self.last_observation = observation
        self.last_action = action
        self.last_reward = reward
        return action

    def reward_giver(self, cooling_equip, clock):
        newpower = cooling_equip.power
        # if clock != self.last_clock:
        #     baseline_power = 0
        #     for machine_state in self.baseline[clock]["cluster_state"]["machine_states"]:
        #         baseline_power += machine_state["power"]
        #     self.last_power = baseline_power
        diff = newpower - self.last_power
        self.last_power = newpower
        self.last_clock = clock
        return -diff

    def reset(self):
        self.last_observation = []
        for _ in range(7):
            self.last_observation.append(0)
        self.last_clock = 0
        self.last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
        with open("./tetris.json")as f:
            self.baseline = json.load(f)


class JSDQNAlgorithm():
    def __init__(self):
        self.dqn = Agent(3 + 4, 1)
        self.steps = 0
        self.last_observation = []
        for _ in range(7):
            self.last_observation.append(0)
        self.last_action = 0
        self.last_reward = 0
        with open("./jobs_info.csv") as f:
            self.jobs_info = f.readline()
        self.jobs_info = self.jobs_info.split(",")
        self.jobs_info = [float(i) for i in self.jobs_info]
        self.jobs_info[4] = 1 if self.jobs_info[4] == 0 else self.jobs_info[4]
        self.jobs_info[6] = 1 if self.jobs_info[6] == 0 else self.jobs_info[6]
        self.jobs_info[8] = 1 if self.jobs_info[8] == 0 else self.jobs_info[8]
        self.jobs_info[10] = 1 if self.jobs_info[10] == 0 else self.jobs_info[10]
        with open("./tetris.json")as f:
            self.baseline = json.load(f)
        self.last_clock = 0
        self.last_power = 0
        self.last_last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0

    def __call__(self, cluster, clock):
        self.total_called_num += 1
        machines = cluster.machines
        tasks = cluster.unfinished_tasks_new.values()
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
                        observation.append((task.task_config.cpu - self.jobs_info[5]) / (self.jobs_info[6] ** 0.5))
                        observation.append((task.task_config.memory - self.jobs_info[7]) / (self.jobs_info[8] ** 0.5))
                        observation.append(
                            (task.task_config.duration - self.jobs_info[9]) / (self.jobs_info[10] ** 0.5))
                        observation.append(
                            (task.task_config.instances_number - self.jobs_info[3]) / (self.jobs_info[4] ** 0.5))
                        observation_list.append(observation)
            if len(observation_list) == 0:
                return None, None, False
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
            return machine, task, done
        return None, None, False

    def reward_giver(self, cluster: Cluster, clock):
        newpower = 0
        for machine in cluster.machines:
            newpower += machine.state["power"]
        if clock != self.last_clock:
            baseline_power = 0
            for machine_state in self.baseline[clock]["cluster_state"]["machine_states"]:
                baseline_power += machine_state["power"]
            self.last_power = baseline_power
        diff = newpower - self.last_power
        self.last_power = newpower
        self.last_clock = clock
        return -diff

    def reset(self):
        self.last_observation = []
        for _ in range(7):
            self.last_observation.append(0)
        self.last_clock = 0
        self.last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
        with open("./tetris.json")as f:
            self.baseline = json.load(f)


class DQNAlgorithm(Algorithm):
    def __init__(self, machine_configs: List[MachineConfig]):
        self.js = JSDQNAlgorithm()
        self.crac = CRACDQNAlgorithm(machine_configs)
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0

    def __call__(self, cluster, clock, cooling_equip=None):
        machine, task, done = self.js(cluster, clock)
        crac_action = self.crac(cluster, task, done, clock, cooling_equip)
        self.total_energy_consume = self.js.total_energy_consume + self.crac.total_energy_consume
        return machine, task, {"Fanspeed": crac_action}

    def reset(self):
        self.js.reset()
        self.crac.reset()
