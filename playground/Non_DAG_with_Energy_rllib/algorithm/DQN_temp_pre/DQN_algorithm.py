from core.algorithm import Algorithm
from core.config import MachineConfig
from core.machine import Machine
from core.cluster import Cluster
from typing import List
from playground.Non_DAG_with_Energy.algorithm.DQN_temp_pre.RL_brain import DeepQNetwork
import numpy as np
from keras.models import load_model


class DQNAlgorithm(Algorithm):
    def __init__(self, machine_configs: List[MachineConfig]):
        self.machine_configs = machine_configs
        self.dqn = DeepQNetwork(n_actions=len(machine_configs),
                                n_features=len(machine_configs) * 2 + 2,
                                learning_rate=0.01, e_greedy=0.9,
                                replace_target_iter=100, memory_size=20000,
                                e_greedy_increment=0.001, batch_size=128)
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
        self.temp_model = load_model(
            "D:/code/CloudSimPy/playground/Non_DAG_with_Energy/algorithm/DQN_temp_pre/predict_temp_usingcpu_ts36_ph12_0.19.hdf5")
        self.cpu_model = load_model(
            "D:/code/CloudSimPy/playground/Non_DAG_with_Energy/algorithm/DQN_temp_pre/predict_cpu_ts36_ph12_0.19.hdf5")

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
                reward = -0.1
                self.total_reward_of_ep += reward
                self.dqn.store_transition(self.last_observation, self.last_action, self.last_reward, observation)
                self.last_observation = observation
                self.last_action = action
                self.last_reward = reward
                if self.steps > 1000:
                    self.dqn.learn()
                self.steps += 1
                return None, None, None
            else:
                pred_temp = self.get_pre_temp(cluster, action)
                machines[action].inlet_temp=pred_temp
                penlty = np.log(1 + np.exp(pred_temp - 26))
                reward = self.reward_giver(cluster, clock) - penlty
                self.total_reward_of_ep += reward
                self.dqn.store_transition(self.last_observation, self.last_action, self.last_reward, observation)
                self.last_observation = observation
                self.last_action = action
                self.last_reward = reward
                if self.steps > 1000 and (self.steps % 100 == 0):
                    self.dqn.learn()
                self.steps += 1
                return machines[action], tasks[0], None
        return None, None, None

    def cal_machine_power(self, machine: Machine):
        cpu_usage = machine.state["cpu_usage_percent"]
        memory_usage = machine.state['memory_usage_percent']
        return 100 * cpu_usage ** 1.9 + 5 * memory_usage

    def get_pre_temp(self, cluster: Cluster, machine_id):
        if len(cluster.monitor.events) < 40:
            return 26
        events = cluster.monitor.events[-36:]
        inlettemp_list = []
        cpu_list = []
        for e in events:
            machine_inlettemp = e['cluster_state']['machine_states'][machine_id]['inlet_temp']
            machine_cpu = e['cluster_state']['machine_states'][machine_id]['cpu_usage_percent']
            inlettemp_list.append(machine_inlettemp)
            cpu_list.append(machine_cpu)
        cpu_list = np.array(cpu_list)
        cpu_list = cpu_list.reshape(-1, cpu_list.shape[0])
        pred_cpu = self.cpu_model.predict(cpu_list)
        pred_cpu=pred_cpu.tolist()[0][0]
        inlettemp_list.append(pred_cpu)
        # print(inlettemp_list)
        inlettemp_list=np.array(inlettemp_list)
        inlettemp_list=inlettemp_list.reshape(-1,inlettemp_list.shape[0])
        pred_temp = self.temp_model.predict(inlettemp_list)
        return pred_temp.tolist()[0][0]

    def reward_giver(self, cluster: Cluster, clock):
        # newpower = 0
        # for machine in cluster.machines:
        #     newpower += self.cal_machine_power(machine)
        # energy_consume = self.last_power * (clock - self.last_clock)
        # self.last_clock = clock
        # self.last_power = newpower
        # self.total_energy_consume += energy_consume
        energy_consume = cluster.monitor.mean_machine_power
        self.total_energy_consume += energy_consume
        if energy_consume == 0:
            return 0
        else:
            return 1 / (energy_consume / 10)

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
