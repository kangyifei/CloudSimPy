import time
import numpy as np
import sys

sys.path.append('..')
from datacentersim.core.config import MachineConfig
from playground.Non_DAG_with_Energy_rllib.utils.csv_reader import CSVReader
from playground.Non_DAG_with_Energy_rllib.utils.tools import multiprocessing_run, average_completion, average_slowdown, \
    average_waiting_time
from playground.Non_DAG_with_Energy_rllib.utils.episode import Episode

from playground.Non_DAG_with_Energy_rllib.algorithm.DQN_rllib_cs.DQN_algorithm import DQNAlgorithm

np.random.seed(41)
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 1
n_iter = 200
jobs_csv = './jobs.csv'
workers_num = 100

machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
jobs_configs = csv_reader.generate(10, jobs_len)
print("-----------------------------------------drl------------------------------------------")

import ray

ray.init(address="auto")


# @ray.remote(num_cpus=2)
class Trainer(object):
    def __init__(self, job_configs):
        machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
        self.episode = Episode(machine_configs,
                               job_configs,
                               DQNAlgorithm(machine_configs),
                               None)

    def train(self):
        for i in range(n_iter):
            print("iter:", i)
            tic = time.time()
            self.episode.run()

            print(self.episode.env.now, time.time() - tic,self.episode.algorithm.total_reward_of_ep)
            self.episode.algorithm.reset()
        return True

trainers = [Trainer(jobs_configs) for _ in range(workers_num)]
res=[t.train() for t in trainers]
# trainers = [Trainer.remote(jobs_configs) for _ in range(workers_num)]
# res=[t.train.remote() for t in trainers]
# print(ray.get(res))
# processes = []
# n_process = 4
# for itr in range(n_process):
#     p = Process(target=train)
#     processes.append(p)
#     p.start()
# for p in processes:
#     p.join()

print("-----------------------------------------test-phase------------------------------------------")
tic = time.time()
jobs_configs = csv_reader.generate(50, 50)
algorithm = DQNAlgorithm(machine_configs)
episode = Episode(machine_configs, jobs_configs, algorithm, "./event_file.json")
episode.run()
print("called nums", algorithm.total_called_num)
print("total energy consume", algorithm.total_energy_consume)
print("total_reward_of_ep", algorithm.total_reward_of_ep)
print(episode.env.now, time.time() - tic, average_completion(episode), average_waiting_time(episode),
      average_slowdown(episode))
algorithm.reset()
