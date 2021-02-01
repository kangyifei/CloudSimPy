
import time
import numpy as np
import sys

sys.path.append('..')

from core.config import MachineConfig
from playground.Non_DAG_with_Energy.algorithm.first_fit import FirstFitTaskalgorithm
from playground.Non_DAG_with_Energy.utils.csv_reader import CSVReader
from playground.Non_DAG_with_Energy.utils.tools import average_completion, average_slowdown, \
    average_waiting_time
from playground.Non_DAG_with_Energy.utils.episode import Episode


np.random.seed(41)

# ************************ Parameters Setting Start ************************
machines_number = 20
jobs_len = 1000
n_iter = 20
n_episode = 12
jobs_csv = './jobs.csv'
if __name__ == '__main__':
    machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
    csv_reader = CSVReader(jobs_csv)
    jobs_configs = csv_reader.generate(0, jobs_len)
    print("-----------------------------------------first_fit------------------------------------------")
    tic = time.time()
    algorithm = FirstFitTaskalgorithm()
    episode = Episode(machine_configs, jobs_configs, algorithm,None)
    episode.run()
    # print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
    print( time.time() - tic)
    print(episode.env.now, average_completion(episode),average_waiting_time(episode), average_slowdown(episode))

