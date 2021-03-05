from core.algorithm import Algorithm
from core.config import MachineConfig
from core.machine import Machine
from core.cluster import Cluster
from typing import List
import numpy as np
from ray.rllib.env.policy_client import PolicyClient
import time
import logging
class DQNAlgorithm(Algorithm):
    def __init__(self, machine_configs: List[MachineConfig]):
        self.machine_configs = machine_configs
        self.steps = 0
        self.last_observation = []
        for machine_config in self.machine_configs:
            self.last_observation.append(machine_config.cpu)
            self.last_observation.append(machine_config.memory)
        self.last_observation.append(0)
        self.last_observation.append(0)
        self.last_action = 0
        self.last_reward = 0

        self.last_clock = 1
        self.last_power = 0
        self.last_last_power=0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
        self.client=PolicyClient("http://192.168.1.100:6521",inference_mode="remote")
        self.eid=self.client.start_episode()
        logger = logging.getLogger("ray")
        logger.setLevel(logging.ERROR)


    def __call__(self, cluster, clock, cooling_equip=None):
        time.sleep(0.001)
        self.total_called_num += 1
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        if not len(tasks) == 0:
            for task in tasks:
                observation = []
                for machine in machines:
                    observation.append(machine.cpu)
                    observation.append(machine.memory)
                observation.append(task.task_config.cpu)
                observation.append(task.task_config.memory)
                observation = np.array(observation)
                action = self.client.get_action(self.eid,observation)
                if action==len(machines):
                    reward = self.reward_giver(cluster, clock)
                    self.total_reward_of_ep += reward
                    self.client.log_returns(self.eid,reward)
                    continue
                if not machines[action].accommodate(task):
                    reward =-0.1
                    self.total_reward_of_ep += reward
                    self.client.log_returns(self.eid, reward)
                    continue
                else:
                    reward = self.reward_giver(cluster, clock)
                    self.total_reward_of_ep += reward
                    self.client.log_returns(self.eid, reward)
                    self.last_observation = observation
                    self.last_action = action
                    self.last_reward = reward
                    return machines[action], task, None
        return None, None, None


    def reward_giver(self, cluster: Cluster, clock):
        newpower = 0
        for machine in cluster.machines:
            newpower += machine.state["power"]
        if newpower==self.last_power:
            #选择不分配
            return 0
        if clock==self.last_clock:
            #计算同一时间步能耗（功率）的增加
            energy_consume=self.last_power-self.last_last_power
        else:
            #计算上一个动作带来的能耗
            energy_consume = self.last_power * (clock - self.last_clock)
            self.total_energy_consume += energy_consume
        # energy_consume = self.last_power - self.last_last_power
        self.last_clock = clock
        self.last_power = newpower
        self.last_last_power=self.last_power
        # power_sum=0
        # machine_state_list = cluster.state['machine_states']
        # for machine_state in machine_state_list:
        #     power_sum += machine_state["power"]
        # energy_consume=cluster.monitor.total_energy_consume-self.total_energy_consume
        # self.total_energy_consume += energy_consume
        if energy_consume == 0:
            return 0
        else:
            return -energy_consume/1000

    def reset(self):
        self.client.end_episode(self.eid,self.last_observation)
        self.last_observation = []
        for machine_config in self.machine_configs:
            self.last_observation.append(machine_config.cpu)
            self.last_observation.append(machine_config.memory)
        self.last_observation.append(0)
        self.last_observation.append(0)
        self.last_clock = 1
        self.last_power = 0
        self.total_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
        self.eid =self.client.start_episode()