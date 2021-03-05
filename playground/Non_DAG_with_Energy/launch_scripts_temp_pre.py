import os
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager
import sys

sys.path.append('..')

from core.config import MachineConfig
from playground.Non_DAG_with_Energy.algorithm.DQN_temp_pre.random_algorithm import RandomTaskalgorithm
from playground.Non_DAG_with_Energy.algorithm.DQN_temp_pre.tetris import Tetris
from playground.Non_DAG_with_Energy.algorithm.DQN_temp_pre.first_fit import FirstFitTaskalgorithm
# from playground.Non_DAG_with_Energy.algorithm.DeepJS.DRL import RLAlgorithm
# from playground.Non_DAG_with_Energy.algorithm.DeepJS.agent import Agent
# from playground.Non_DAG_with_Energy.algorithm.DeepJS.brain import Brain
#
# from playground.Non_DAG_with_Energy.algorithm.DeepJS.reward_giver import AverageCompletionRewardGiver

from playground.Non_DAG_with_Energy.utils.csv_reader import CSVReader
from playground.Non_DAG_with_Energy.utils.feature_functions import features_extract_func_ac, features_normalize_func_ac
from playground.Non_DAG_with_Energy.utils.tools import multiprocessing_run, average_completion, average_slowdown, \
    average_waiting_time
from playground.Non_DAG_with_Energy.utils.episode import Episode
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# tf.compat.v1.disable_eager_execution()

np.random.seed(41)
tf.random.set_seed(1)
tf.compat.v1.disable_eager_execution()
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 10
n_iter = 100
n_episode = 12
jobs_csv = './jobs.csv'

machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
jobs_configs = csv_reader.generate(0, jobs_len)
import playground.Non_DAG_with_Energy.algorithm.DQN_temp_pre.DQN_algorithm as dqn_tp

algorithm = dqn_tp.DQNAlgorithm(machine_configs)
total_energy_consume_list = []
total_called_num_list = []
total_reward_of_ep_list = []
average_completion_list = []
average_waitting_time_list = []
for index in range(n_iter):
    print("iter:", index)
    tic = time.time()
    episode = Episode(machine_configs, jobs_configs, algorithm, "./event_file.json")
    episode.run()
    total_called_num_list.append(algorithm.total_called_num)
    print("called nums", algorithm.total_called_num)
    total_energy_consume_list.append(algorithm.total_energy_consume)
    print("total energy consume", algorithm.total_energy_consume)
    total_reward_of_ep_list.append(algorithm.total_reward_of_ep)
    print("total_reward_of_ep", algorithm.total_reward_of_ep)
    average_completion_res = average_completion(episode)
    average_completion_list.append(average_completion_res)
    average_waitting_time_res = average_waiting_time(episode)
    average_waitting_time_list.append(average_waitting_time_res)
    print(episode.env.now, time.time() - tic, average_completion_res, average_waitting_time_res,
          average_slowdown(episode))
    if (index % 10 == 0):
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
    algorithm.reset()
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
jobs_configs = csv_reader.generate(10, 10)
episode = Episode(machine_configs, jobs_configs, algorithm, "./test_event_file_pre.json")
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
print("-----------------------------------------tetris------------------------------------------")
tic = time.time()
algorithm = Tetris()
episode = Episode(machine_configs, jobs_configs, algorithm, "./tetris.json")
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
plt.xlabel("tetris")
plt.show()
print(machine_inlet_list)
print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),
      average_slowdown(episode))
print("-----------------------------------------first_fit------------------------------------------")
tic = time.time()
algorithm = FirstFitTaskalgorithm()
episode = Episode(machine_configs, jobs_configs, algorithm, "./tetris.json")
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
plt.xlabel("first_fit")
plt.show()
print(machine_inlet_list)
print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),
      average_slowdown(episode))
print("-----------------------------------------random------------------------------------------")
random_ec = []
random_ac = []
random_wt = []
for i in range(1):
    tic = time.time()
    algorithm = RandomTaskalgorithm()
    episode = Episode(machine_configs, jobs_configs, algorithm, "./tetris.json")
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
    plt.xlabel("random")
    plt.show()
    print(machine_inlet_list)
    # print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
    random_ec.append(episode.simulation.monitor[0].total_energy_consume)
    random_ac.append(average_completion(episode))
    random_wt.append(average_waiting_time(episode))
    # print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),average_slowdown(episode))
# print(random_ec)
# print(random_ac)
# print("min_random_ec", random_ec[np.argmin(np.array(random_ec))])
# print("min_random_ac", random_ac[np.argmin(np.array(random_ac))])
# print("min_random_wt", random_ac[np.argmin(np.array(random_wt))])
# print("mean_random_ec", np.average(np.array(random_ec)))
# print("mean_random_ac", np.average(np.array(random_ac)))
# print("mean_random_wt", np.average(np.array(random_wt)))

#
# for itr in range(n_iter):
#     tic = time.time()
#     print("********** Iteration %i ************" % itr)
#     processes = []
#
#     manager = Manager()
#     trajectories = manager.list([])
#     makespans = manager.list([])
#     average_completions = manager.list([])
#     average_slowdowns = manager.list([])
#     for i in range(n_episode):
#         algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
#                                 features_normalize_func=features_normalize_func)
#         episode = Episode(machine_configs, jobs_configs, algorithm, None)
#         algorithm.reward_giver.attach(episode.simulation)
#         p = Process(target=multiprocessing_run,
#                     args=(episode, trajectories, makespans, average_completions, average_slowdowns))
#
#         processes.append(p)
#
#     for p in processes:
#         p.start()
#
#     for p in processes:
#         p.join()
#
#     agent.log('makespan', np.mean(makespans), agent.global_step)
#     agent.log('average_completions', np.mean(average_completions), agent.global_step)
#     agent.log('average_slowdowns', np.mean(average_slowdowns), agent.global_step)
#
#     toc = time.time()
#
#     print(np.mean(makespans), toc - tic, np.mean(average_completions), np.mean(average_slowdowns))
#
#     all_observations = []
#     all_actions = []
#     all_rewards = []
#     for trajectory in trajectories:
#         observations = []
#         actions = []
#         rewards = []
#         for node in trajectory:
#             observations.append(node.observation)
#             actions.append(node.action)
#             rewards.append(node.reward)
#
#         all_observations.append(observations)
#         all_actions.append(actions)
#         all_rewards.append(rewards)
#
#     all_q_s, all_advantages = agent.estimate_return(all_rewards)
#
#     agent.update_parameters(all_observations, all_actions, all_advantages)
#
# agent.save()
