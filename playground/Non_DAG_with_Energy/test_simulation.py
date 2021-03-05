import time
import numpy as np
import sys

sys.path.append('C:\\code\\CloudSimPy\\CloudSimPy\\')

from core.config import MachineConfig
from playground.Non_DAG_with_Energy.algorithm.first_fit import FirstFitTaskalgorithm
from playground.Non_DAG_with_Energy.algorithm.first_fit_new import NewFirstFitTaskalgorithm
from playground.Non_DAG_with_Energy.utils.csv_reader import CSVReader
from playground.Non_DAG_with_Energy.utils.tools import average_completion, average_slowdown, \
    average_waiting_time
from playground.Non_DAG_with_Energy.utils.episode import Episode

np.random.seed(41)

# ************************ Parameters Setting Start ************************
machines_number = 5
# jobs_len = 10
jobs_csv = './jobs.csv'
machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
n_jiasubi=[]
for _ in range(10):
    jiasubi = []
    for jobs_len in range(1, 101):
        jobs_configs = csv_reader.generate(0, jobs_len)
        print("-----------------------------------------first_fit------------------------------------------")
        tic = time.time()
        algorithm = FirstFitTaskalgorithm()
        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        episode.run()
        # print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
        old = time.time() - tic
        print(old)
        # print(episode.env.now, average_completion(episode), average_waiting_time(episode), average_slowdown(episode))
        print("-----------------------------------------new_first_fit------------------------------------------")
        tic = time.time()
        algorithm = NewFirstFitTaskalgorithm()
        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        episode.run()
        # print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
        new = time.time() - tic
        print(new)
        jsb = (old - new) / old * 100
        jiasubi.append(jsb)
        print("%:", (old - new) / old * 100)
        # print(episode.env.now, average_completion(episode), average_waiting_time(episode), average_slowdown(episode))
    n_jiasubi.append(jiasubi)
import numpy as np
n_jiasubi=np.array(n_jiasubi)
np.save("./jiasubi.npy",n_jiasubi)

# import matplotlib.pyplot as plt
#
# plt.plot(np.arange(len(jiasubi)), jiasubi)
# plt.xlabel('新方法加速比')
# plt.savefig("./test_new_unfinished_task.png")
# plt.show()
# print(jiasubi)
