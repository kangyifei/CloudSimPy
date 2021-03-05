import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager, freeze_support
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
import torch
from core.config import MachineConfig
from playground.Non_DAG_with_Energy_rllib.algorithm.random_algorithm import RandomTaskalgorithm
from playground.Non_DAG_with_Energy_rllib.algorithm.tetris import Tetris
from playground.Non_DAG_with_Energy_rllib.algorithm.first_fit import FirstFitTaskalgorithm
from playground.Non_DAG_with_Energy_rllib.utils.csv_reader import CSVReader
from playground.Non_DAG_with_Energy_rllib.utils.tools import multiprocessing_run, average_completion, average_slowdown, \
    average_waiting_time
from playground.Non_DAG_with_Energy_rllib.utils.episode import Episode

from playground.Non_DAG_with_Energy_rllib.algorithm.DQN_rllib.DQN_algorithm import DQNAlgorithm

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray import tune

np.random.seed(41)
tf.random.set_seed(1)
torch.manual_seed(1)
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 2
n_iter = 2000
jobs_csv = './jobs.csv'

machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
jobs_configs = csv_reader.generate(10, jobs_len)
algorithm = DQNAlgorithm(machine_configs)
print("-----------------------------------------drl------------------------------------------")
total_energy_consume_list = []
total_called_num_list = []
total_reward_of_ep_list = []
average_completion_list = []
average_waitting_time_list = []
ray.init()
# failed-1
# config=ray.rllib.agents.dqn.DEFAULT_CONFIG.copy()
# config["num_gpus"]=1
# config["num_workers"]=1
# from ray.tune.registry import register_env
# from ray.tune.logger import pretty_print
# register_env("cloudsim",lambda config:Episode(config))
# trainer=DQNTrainer(config=config,env="cloudsim")

# for i in range(100):
#     res=trainer.train()
#     print(pretty_print(res))

# failed-2
# tune.run(DQNTrainer,config=config)

import gym
from gym import spaces
import random


class CloudSimStub(gym.Env):
    def __init__(self, config):
        self.__init_observation_action_space__(machine_configs, None)

    def __init_observation_action_space__(self,
                                          machines_config_list,
                                          cooling_equipment_config=None) -> None:
        # add machines params
        box_high_bound = []
        box_low_bound = []
        for machine_config in machines_config_list:
            box_low_bound.append(0)
            box_high_bound.append(machine_config.cpu_capacity)
            box_low_bound.append(0)
            box_high_bound.append(machine_config.memory_capacity)
        # add cooling equipments params
        if cooling_equipment_config is not None:
            state_paramslist = cooling_equipment_config.state_paramslist
            for state_param_key in state_paramslist:
                box_low_bound.append(state_paramslist[state_param_key]["low"])
                box_high_bound.append(state_paramslist[state_param_key]["high"])
            # control_paramslist = cooling_equipment_config.control_paramslist
            # for control_param_key in control_paramslist:
            #     temp_action_space.append(spaces.Box(low=control_paramslist[control_param_key]["low"],
            #                                         high=control_paramslist[control_param_key]["high"],
            #                                         shape=(1,)))
        # add coming task params including cpu,mem,disk
        maxmum_cpu_capacity = max([machine_config.cpu_capacity for machine_config in machines_config_list])
        maxmum_mem_capacity = max([machine_config.memory_capacity for machine_config in machines_config_list])
        maxmum_disk_capacity = max([machine_config.disk_capacity for machine_config in machines_config_list])
        box_low_bound.append(0)
        box_high_bound.append(maxmum_cpu_capacity)
        box_low_bound.append(0)
        box_high_bound.append(maxmum_mem_capacity)
        box_low_bound = np.array(box_low_bound)
        box_high_bound = np.array(box_high_bound)
        self.observation_space = spaces.Box(box_low_bound, box_high_bound)
        self.action_space = spaces.Discrete(len(machines_config_list) + 1)
        self.steps = 0

    def reset(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        done = False
        return self.observation_space.sample(), \
               float(random.random), \
               done, \
               {}


from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
register_env("cloudsim",lambda config:CloudSimStub(config))
SERVER_ADDRESS = "127.0.0.1"
PORT = 6521
trainer_config = {
    "input": (
        lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, PORT)),
    # Use a single worker process (w/ SyncSampler) to run the server.
    "num_workers": 0,
    # Disable OPE, since the rollouts are coming from online clients.
    "input_evaluation": [],
}
trainer=DQNTrainer(config=trainer_config,env="cloudsim")
for i in range(n_iter):
    # try:
    print("iter:", i)
    tic = time.time()
    episode = Episode(machine_configs, jobs_configs, algorithm, "./event_file.json")
    if __name__ == "__main__":
        freeze_support()
        episode.run()
    total_called_num_list.append(algorithm.total_called_num)
    print("called nums", algorithm.total_called_num)
    total_energy_consume_list.append(episode.simulation.cluster.monitor.total_energy_consume)
    print("total energy consume", episode.simulation.cluster.monitor.total_energy_consume)
    total_reward_of_ep_list.append(algorithm.total_reward_of_ep)
    print("total_reward_of_ep", algorithm.total_reward_of_ep)
    average_completion_res = average_completion(episode)
    average_completion_list.append(average_completion_res)
    average_waitting_time_res = average_waiting_time(episode)
    average_waitting_time_list.append(average_waitting_time_res)
    print(episode.env.now, time.time() - tic, average_completion_res, average_waitting_time_res,
          average_slowdown(episode))
    algorithm.reset()
    if (i % 100 == 0 and i != 0):
        # plot reward of ep per 100 cycles
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(len(total_reward_of_ep_list)),
                total_reward_of_ep_list, 'g-'
                )
        # plt.xlabel('l1:total_reward_of_ep_'+str(i))
        # plt.savefig("./total_reward_of_ep.png")
        # plt.show()
        ax1 = ax.twinx()
        ax1.plot(np.arange(len(total_energy_consume_list)),
                 total_energy_consume_list, 'b--'
                 )
        ax2 = ax.twinx()
        ax2.plot(np.arange(len(total_called_num_list)),
                 total_called_num_list, 'r--'
                 )
        ax.set_ylabel('reward', color='g')
        ax1.set_ylabel('energy', color='b')  # 设置Y1轴标题
        ax2.set_ylabel('call_nums', color='r')
        ax1.set_xlabel('ep_' + str(i))
        plt.savefig("./energyconsume.png")
        plt.show()
        # algorithm.dqn.plot_cost()
    # except BaseException as e:
    #     print(e)
print(total_energy_consume_list)
print(total_called_num_list)
print(average_completion_list)

plt.plot(np.arange(len(total_called_num_list)), total_called_num_list)
plt.xlabel('l1:called num')
plt.savefig("./callednum.png")
plt.show()
plt.plot(np.arange(len(total_energy_consume_list)), total_energy_consume_list)
plt.xlabel('l1:energy_consume')
plt.savefig("./energyconsume.png")
plt.show()
plt.plot(np.arange(len(total_reward_of_ep_list)), total_reward_of_ep_list)
plt.xlabel('l1:total_reward_of_ep')
plt.savefig("./total_reward_of_ep.png")
plt.show()
plt.plot(np.arange(len(average_completion_list)), average_completion_list)
plt.xlabel('l1:average_completion')
plt.savefig("./average_completion_list.png")
plt.show()
plt.plot(np.arange(len(average_waitting_time_list)), average_waitting_time_list)
plt.xlabel('l1:average_waitting_time_list')
plt.savefig("./average_waitting_time_list.png")
plt.show()
energy_consume_divided_by_reward = []
for i in range(len(total_energy_consume_list)):
    energy_consume_divided_by_reward.append(total_energy_consume_list[i] / total_reward_of_ep_list[i])
plt.plot(np.arange(len(energy_consume_divided_by_reward)), energy_consume_divided_by_reward)
plt.xlabel('l1:energy_consume/total_reward_of_ep')
plt.show()
algorithm.dqn.plot_cost()

print("-----------------------------------------test-phase------------------------------------------")
tic = time.time()
jobs_configs = csv_reader.generate(50, 50)
episode = Episode(machine_configs, jobs_configs, algorithm, "./event_file.json")
episode.run()
print("called nums", algorithm.total_called_num)
print("total energy consume", algorithm.total_energy_consume)
print("total_reward_of_ep", algorithm.total_reward_of_ep)
print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),
      average_slowdown(episode))
algorithm.reset()
