from core.algorithm import Algorithm
from core.config import MachineConfig
from core.machine import Machine
from core.cluster import Cluster
from typing import List
from .RL_brain import DeepQNetwork, PolicyGradient
import numpy as np
from keras.models import load_model


class DQNAlgorithm(Algorithm):
    def __init__(self, machine_configs: List[MachineConfig]):
        self.machine_configs = machine_configs
        self.task_dqn = DeepQNetwork(scope='task', n_actions=len(machine_configs),
                                     n_features=len(machine_configs) * 2 + 2,
                                     learning_rate=0.01, e_greedy=0.9,
                                     replace_target_iter=100, memory_size=20000,
                                     e_greedy_increment=0.001, )
        self.CRAC_dqn = DeepQNetwork(scope='crac', n_actions=(30 - 0) * 10,
                                     n_features=len(machine_configs) * 3 + 2,
                                     learning_rate=0.01, e_greedy=0.9,
                                     replace_target_iter=100, memory_size=20000,
                                     e_greedy_increment=0.001, )

        # self.CRAC_dqn = PolicyGradient(n_actions=(30 - 0) * 10,
        #                                n_features=len(machine_configs) * 3 + 2,
        #                                learning_rate=0.02,
        #                                reward_decay=0.99, )
        self.steps = 0
        self.task_last_observation = []
        for machine_config in self.machine_configs:
            self.task_last_observation.append(machine_config.cpu)
            self.task_last_observation.append(machine_config.memory)
        self.task_last_observation.append(0)
        self.task_last_observation.append(0)
        self.task_last_action = 0
        self.task_last_reward = 0

        self.CRAC_last_observation = []
        for machine_config in self.machine_configs:
            self.CRAC_last_observation.append(machine_config.cpu)
            self.CRAC_last_observation.append(machine_config.memory)
            self.CRAC_last_observation.append(26)
        self.CRAC_last_observation.append(0)
        self.CRAC_last_observation.append(0)
        self.CRAC_last_action = 0
        self.CRAC_last_reward = 0

        self.last_clock = 0
        self.last_power = 0
        self.total_energy_consume = 0
        self.crac_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
        self.temp_model = load_model(
            "D:\code\CloudSimPy\core\prediction_model\model\predict_temp_usingcpu+condition_ts36_ph12_0.16.hdf5")
        self.cpu_model = load_model(
            "D:/code/CloudSimPy/playground/Non_DAG_with_Energy/algorithm/DQN_temp_pre/predict_cpu_ts36_ph12_0.19.hdf5")

    def __call__(self, cluster, clock, cooling_equip=None):
        self.total_called_num += 1
        tasks = cluster.tasks_which_has_waiting_instance
        task_observation, CRAC_observation = self.get_observation(cluster)
        if not len(tasks) == 0:
            paramlist = self.CRAC_sche(cluster, clock, CRAC_observation)
            choosed_machine, choosed_task = self.task_sche(cluster, clock, task_observation,
                                                           conditon_set_point=paramlist["set_temp"])
            return choosed_machine, choosed_task, paramlist
        else:
            paramlist = self.CRAC_sche(cluster, clock, CRAC_observation)
            return None, None, paramlist

    def CRAC_sche(self, cluster, clock, observation):
        action = self.CRAC_dqn.choose_action(observation)
        reward = self.crac_reward_giver(cluster)
        self.total_reward_of_ep += reward
        self.CRAC_dqn.store_transition(self.CRAC_last_observation, self.CRAC_last_action, self.CRAC_last_reward,
                                       observation)
        self.CRAC_last_observation = observation
        self.CRAC_last_action = action
        self.CRAC_last_reward = reward
        if self.steps > 1000 and (self.steps % 500 == 0):
            self.CRAC_dqn.learn()
        self.steps += 1
        return {"set_temp": action / 10}

    def task_sche(self, cluster, clock, observation, conditon_set_point):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        action = self.task_dqn.choose_action(observation)
        if not machines[action].accommodate(tasks[0]):
            reward = -0.1
            self.total_reward_of_ep += reward
            self.task_dqn.store_transition(self.task_last_observation, self.task_last_action, self.task_last_reward,
                                           observation)
            self.task_last_observation = observation
            self.task_last_action = action
            self.task_last_reward = reward
            if self.steps > 1000:
                self.task_dqn.learn()
            self.steps += 1
            return None, None
        else:
            pred_temp = self.get_pre_temp(cluster, action, conditon_set_point)
            machines[action].inlet_temp = pred_temp
            penlty = np.log(1 + np.exp(pred_temp - 26))
            reward = self.task_reward_giver(cluster) - penlty
            self.total_reward_of_ep += reward
            self.task_dqn.store_transition(observation, action, reward, self.task_last_observation)
            self.task_last_observation = observation
            if self.steps > 1000 and (self.steps % 100 == 0):
                self.task_dqn.learn()
            self.steps += 1
            return machines[action], tasks[0]

    def get_pre_temp(self, cluster: Cluster, machine_id, condition_set_temp):
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
        pred_cpu = pred_cpu.tolist()[0][0]
        inlettemp_list.append(condition_set_temp)
        inlettemp_list.append(pred_cpu)
        # print(inlettemp_list)
        inlettemp_list = np.array(inlettemp_list)
        inlettemp_list = inlettemp_list.reshape(-1, inlettemp_list.shape[0])
        pred_temp = self.temp_model.predict(inlettemp_list)
        return pred_temp.tolist()[0][0]

    def get_observation(self, cluster):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        cpu_observation = []
        memory_observation = []
        inlet_observation = []
        for machine in machines:
            cpu_observation.append(machine.cpu)
            memory_observation.append(machine.memory)
            inlet_observation.append(machine.inlet_temp)
        cpu_observation, memory_observation, inlet_observation = np.array(cpu_observation), np.array(
            memory_observation), np.array(inlet_observation)
        task_observation = np.dstack((cpu_observation, memory_observation))
        CRAC_obervation = np.dstack((cpu_observation, memory_observation, inlet_observation))
        task_observation = task_observation.reshape(-1)
        CRAC_obervation = CRAC_obervation.reshape(-1)
        if len(tasks) != 0:
            new_task_ob = np.array([tasks[0].task_config.cpu, tasks[0].task_config.memory])
        else:
            new_task_ob = np.array([0, 0])
        task_observation = np.concatenate((task_observation, new_task_ob))
        CRAC_obervation = np.concatenate((CRAC_obervation, new_task_ob))
        return task_observation, CRAC_obervation

    def crac_reward_giver(self, cluster: Cluster):
        energy_consume = cluster.monitor.crac_power
        self.crac_energy_consume += energy_consume
        if energy_consume == 0:
            return 0
        else:
            return 1 / (energy_consume / 1000)

    def task_reward_giver(self, cluster: Cluster):
        energy_consume = cluster.monitor.mean_machine_power
        self.total_energy_consume += energy_consume
        if energy_consume == 0:
            return 0
        else:
            return 1 / (energy_consume / 10)

    def reset(self):
        self.task_last_observation = []
        for machine_config in self.machine_configs:
            self.task_last_observation.append(machine_config.cpu)
            self.task_last_observation.append(machine_config.memory)
        self.task_last_observation.append(0)
        self.task_last_observation.append(0)
        self.task_last_action = 0
        self.task_last_reward = 0

        self.CRAC_last_observation = []
        for machine_config in self.machine_configs:
            self.CRAC_last_observation.append(machine_config.cpu)
            self.CRAC_last_observation.append(machine_config.memory)
            self.CRAC_last_observation.append(26)
        self.CRAC_last_observation.append(0)
        self.CRAC_last_observation.append(0)
        self.CRAC_last_action = 0
        self.CRAC_last_reward = 0

        self.last_clock = 0
        self.last_power = 0
        self.total_energy_consume = 0
        self.crac_energy_consume = 0
        self.total_called_num = 0
        self.total_reward_of_ep = 0
