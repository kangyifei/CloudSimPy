
import time
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
import torch
from core.config import MachineConfig
from playground.Non_DAG_with_Energy.algorithm.random_algorithm import RandomTaskalgorithm
from playground.Non_DAG_with_Energy.algorithm.tetris import Tetris
from playground.Non_DAG_with_Energy.algorithm.first_fit import FirstFitTaskalgorithm

from playground.Non_DAG_with_Energy.utils.csv_reader import CSVReader
from playground.Non_DAG_with_Energy.utils.tools import multiprocessing_run, average_completion, average_slowdown, \
    average_waiting_time
from playground.Non_DAG_with_Energy.utils.episode import Episode

from playground.Non_DAG_with_Energy.algorithm.dueling_DDQN.DQN_algorithm import DQNAlgorithm


# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 10
n_iter = 2000
jobs_csv = "./jobs.csv"
machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
jobs_configs = csv_reader.generate(0, jobs_len)
print("-----------------------------------------first_fit------------------------------------------")
tic = time.time()
algorithm = FirstFitTaskalgorithm()
episode = Episode(machine_configs, jobs_configs, algorithm, "./first_fit.json")
episode.run()
print("total energy consume", episode.simulation.monitor[0].total_energy_consume)
print(episode.env.now, time.time() - tic, average_completion(episode),average_waiting_time(episode), average_slowdown(episode))
