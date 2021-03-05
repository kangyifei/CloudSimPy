import os
import time
import numpy as np
# import tensorflow as tf
from multiprocessing import Process, Manager, freeze_support
import sys
import matplotlib.pyplot as plt
import math

sys.path.append('..')
import torch
from core.config import MachineConfig
from playground.Non_DAG_with_Energy.algorithm.random_algorithm import RandomTaskalgorithm
from playground.Non_DAG_with_Energy.algorithm.tetris import Tetris
from playground.Non_DAG_with_Energy.algorithm.first_fit import FirstFitTaskalgorithm
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

from playground.Non_DAG_with_Energy.algorithm.dueling_DDQN_take_couple_comparedto_tetris.DQN_algorithm import \
    DQNAlgorithm
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# tf.compat.v1.disable_eager_execution()

np.random.seed(41)
# tf.random.set_seed(1)
torch.manual_seed(1)
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 15
n_iter = 400
jobs_csv = './jobs.csv'
machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
dqn_algorithm = None

total_energy_consume_list = []
total_called_num_list = []
total_reward_of_ep_list = []
average_completion_list = []
average_waitting_time_list = []
jobs_len_now_last=0
for i in range(n_iter):
    # try:
    jobs_len_now = int((jobs_len / n_iter) * i)
    jobs_len_now = 1 if jobs_len_now == 0 else jobs_len_now
    jobs_configs = csv_reader.generate(0, jobs_len_now)
    print("iter:", i)
    print("jobs_len_now:", jobs_len_now)
    print("-----------------------------------------first_fit------------------------------------------")
    tic = time.time()
    algorithm = FirstFitTaskalgorithm()
    episode = Episode(machine_configs, jobs_configs, algorithm, "./tetris_jianjin.json")
    with open("./pickle_test","bw+") as f:
        pickle.dump(episode,f)