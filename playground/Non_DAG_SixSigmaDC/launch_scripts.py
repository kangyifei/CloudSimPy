import os
import time
import numpy as np
import tensorflow as tf
import sys
sys.path.append('..')

from core.config import MachineConfig, CoolingEquipmentConfig
from core.SixSigmaDC.Room import Room

from playground.Non_DAG_SixSigmaDC.utils.csv_reader import CSVReader
from playground.Non_DAG_SixSigmaDC.utils.tools import average_completion, average_slowdown, average_waiting_time
from playground.Non_DAG_SixSigmaDC.utils.episode import Episode
from playground.Non_DAG_SixSigmaDC.algorithm.DQN_Policy_gradient_CRAC.DQN_algorithm import DQNAlgorithm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.compat.v1.disable_eager_execution()

np.random.seed(41)
# tf.random.set_random_seed(41)
# ************************ Parameters Setting Start ************************
machines_number = 50
jobs_len = 50
n_iter = 1
jobs_csv = './jobs.csv'

# brain = Brain(9)
# reward_giver = AverageCompletionRewardGiver()
# features_extract_func = features_extract_func_ac
# features_normalize_func = features_normalize_func_ac
#
# model_dir = './agents/%s' % name
# # ************************ Parameters Setting End ************************
#
# if not os.path.isdir(model_dir):
#     os.makedirs(model_dir)
#
# agent = Agent(name, brain, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
#               model_save_path='%s/model.ckpt' % model_dir)
import matplotlib.pyplot as plt

machine_configs = [MachineConfig(32, 1, 1) for i in range(machines_number)]
CRAC_config = CoolingEquipmentConfig("../../core/CRAC/CRAC.json")
csv_reader = CSVReader(jobs_csv)
jobs_configs = csv_reader.generate(0, jobs_len)
base_path= "D:\code\CloudSimPy\playground\\Non_DAG_SixSigmaDC\\BigServer\\1-\SolverExchange"
TenServerRoom = Room(base_path)
algorithm = DQNAlgorithm(machine_configs)
total_energy_consume_list = []
total_called_num_list = []
total_reward_of_ep_list = []
for index in range(n_iter):
    print("iter:", index)
    tic = time.time()
    episode = Episode(machine_configs, jobs_configs, algorithm, "./event_file.json", TenServerRoom)
    episode.run()
    monitor = episode.simulation.cluster.monitor
    events = monitor.events
    machine_inlet_list = []
    for i in range(len(machine_configs)):
        machine_inlet_list.append([])
    for e in events:
        for i in range(len(machine_configs)):
            machine_inlet_list[i].append(e['cluster_state']['machine_states'][i]['inlet_temp'])
    for i in range(len(machine_configs)):
        plt.plot(np.arange(len(machine_inlet_list[i])), machine_inlet_list[i], label=str(i) + "st machine")
    plt.legend()
    plt.xlabel("iter" + str(index) + ": machine inlettemp")
    plt.show()
    total_called_num_list.append(algorithm.total_called_num)
    print("called nums", algorithm.total_called_num)
    total_energy_consume_list.append(algorithm.total_energy_consume)
    print("total energy consume", algorithm.total_energy_consume)
    total_reward_of_ep_list.append(algorithm.total_reward_of_ep)
    print("total_reward_of_ep", algorithm.total_reward_of_ep)
    print("total_crac_energy_consume", algorithm.crac_energy_consume)
    print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
    #等待6SigmaDC运行完毕

    plt.plot(np.arange(len(total_called_num_list)), total_called_num_list)
    plt.xlabel('called num')
    plt.title("iter:" + str(index))
    plt.savefig("./callednum_iter"+ str(index)+".png")
    plt.show()

    plt.plot(np.arange(len(total_energy_consume_list)), total_energy_consume_list)
    plt.xlabel('energy_consume')
    plt.title("iter:" + str(index))
    plt.savefig("./energyconsume_iter"+ str(index)+".png")
    plt.show()

    plt.plot(np.arange(len(total_reward_of_ep_list)), total_reward_of_ep_list)
    plt.xlabel('total_reward_of_ep')
    plt.title("iter:" + str(index))
    plt.savefig("./total_reward_of_ep_iter"+ str(index)+".png")
    plt.show()

    energy_consume_divided_by_reward = []
    for i in range(len(total_energy_consume_list)):
        energy_consume_divided_by_reward.append(total_energy_consume_list[i] / total_reward_of_ep_list[i])
    plt.plot(np.arange(len(energy_consume_divided_by_reward)), energy_consume_divided_by_reward)
    plt.title("iter:"+str(index))
    plt.xlabel('energy_consume/total_reward_of_ep')
    plt.show()

    algorithm.task_dqn.plot_cost()
    algorithm.CRAC_dqn.plot_cost()
    algorithm.reset()

    TenServerRoom.stop()

# print(total_energy_consume_list)
# print(total_called_num_list)



print("-----------------------------------------test-phase------------------------------------------")
tic = time.time()
jobs_configs = csv_reader.generate(10, 10)
episode = Episode(machine_configs, jobs_configs, algorithm, "./test_event_file_pre.json", CRAC_config)
episode.run()
monitor = episode.simulation.cluster.monitor
events = monitor.events
machine_inlet_list = []
for i in range(len(machine_configs)):
    machine_inlet_list.append([])
for e in events:
    for i in range(len(machine_configs)):
        machine_inlet_list[i].append(e['cluster_state']['machine_states'][i]['inlet_temp'])
for i in range(len(machine_configs)):
    plt.plot(np.arange(len(machine_inlet_list[i])), machine_inlet_list[i], label=str(i) + "st machine")
plt.legend()
plt.xlabel("temp_pre_test_machine_inlet_temp")
plt.show()
print(machine_inlet_list)
print("called nums", algorithm.total_called_num)
print("total energy consume", algorithm.total_energy_consume)
print("total_reward_of_ep", algorithm.total_reward_of_ep)
print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),
      average_slowdown(episode))
algorithm.reset()
# print("-----------------------------------------tetris------------------------------------------")
# tic = time.time()
# algorithm = Tetris()
# episode = Episode(machine_configs, jobs_configs, algorithm, "./tetris.json",CRAC_config)
# episode.run()
# monitor = episode.simulation.cluster.monitor
# events = monitor.events
# machine_inlet_list = []
# for i in range(len(machine_configs)):
#     machine_inlet_list.append([])
# for e in events:
#     for i in range(len(machine_configs)):
#         machine_inlet_list[i].append(e['cluster_state']['machine_states'][i]['inlet_temp'])
# for i in range(len(machine_configs)):
#     plt.plot(np.arange(len(machine_inlet_list[i])), machine_inlet_list[i], label=str(i) + "st machine")
# plt.legend()
# plt.xlabel("tetris")
# plt.show()
# print(machine_inlet_list)
# print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
# print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),
#       average_slowdown(episode))
# print("-----------------------------------------first_fit------------------------------------------")
# tic = time.time()
# algorithm = FirstFitTaskalgorithm()
# episode = Episode(machine_configs, jobs_configs, algorithm, "./tetris.json",CRAC_config)
# episode.run()
# monitor = episode.simulation.cluster.monitor
# events = monitor.events
# machine_inlet_list = []
# for i in range(len(machine_configs)):
#     machine_inlet_list.append([])
# for e in events:
#     for i in range(len(machine_configs)):
#         machine_inlet_list[i].append(e['cluster_state']['machine_states'][i]['inlet_temp'])
# for i in range(len(machine_configs)):
#     plt.plot(np.arange(len(machine_inlet_list[i])), machine_inlet_list[i], label=str(i) + "st machine")
# plt.legend()
# plt.xlabel("first_fit")
# plt.show()
# print(machine_inlet_list)
# print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
# print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),
#       average_slowdown(episode))
# print("-----------------------------------------random------------------------------------------")
# random_ec = []
# random_ac = []
# random_wt = []
# for i in range(1):
#     tic = time.time()
#     algorithm = RandomTaskalgorithm()
#     episode = Episode(machine_configs, jobs_configs, algorithm, "./tetris.json",CRAC_config)
#     episode.run()
#     monitor = episode.simulation.cluster.monitor
#     events = monitor.events
#     machine_inlet_list = []
#     for i in range(len(machine_configs)):
#         machine_inlet_list.append([])
#     for e in events:
#         for i in range(len(machine_configs)):
#             machine_inlet_list[i].append(e['cluster_state']['machine_states'][i]['inlet_temp'])
#     for i in range(len(machine_configs)):
#         plt.plot(np.arange(len(machine_inlet_list[i])), machine_inlet_list[i], label=str(i) + "st machine")
#     plt.legend()
#     plt.xlabel("random")
#     plt.show()
#     print(machine_inlet_list)
#     # print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
#     random_ec.append(episode.simulation.monitor[0].total_energy_consume)
#     random_ac.append(average_completion(episode))
#     random_wt.append(average_waiting_time(episode))
