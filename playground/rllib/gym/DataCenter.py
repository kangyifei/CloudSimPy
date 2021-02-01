from playground.rllib.utils.csv_reader import CSVReader
from playground.rllib.utils.episode import Episode
from typing import List
from core.config import MachineConfig
from core.cluster import Cluster
from core.broker import Broker
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
import simpy
import numpy as np
class DataCenter(ExternalMultiAgentEnv):
    def __init__(self, action_space, observation_space, max_concurrent=100,jobs_csv=None,jobs_len=10,jobs_offset=0,machines_configs=None):
        super().__init__(action_space, observation_space, max_concurrent)
        self.jobs_len=jobs_len
        self.machine_configs=machines_configs
        self.jobs_csv = jobs_csv
        self.jobs_offset=jobs_offset
        if self.jobs_csv is None:
            raise ValueError("jobs file must exists")
        if self.machine_configs is None:
            raise ValueError("machine configs must exists")
        elif self.machine_configs is not List[MachineConfig]:
            raise ValueError("machine configs type error")

    def run(self):

        pass
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
            
            action = self.get_action(self.episode_id,observation)
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
                reward = self.reward_giver(cluster, clock)
                self.total_reward_of_ep += reward
                self.dqn.store_transition(self.last_observation, self.last_action, self.last_reward, observation)
                self.last_observation = observation
                self.last_action = action
                self.last_reward = reward
                if self.steps > 1000 and (self.steps % 500 == 0):
                    self.dqn.learn()
                self.steps += 1
                return machines[action], tasks[0], None
        return None, None, None
    def start_episode(self, episode_id=None, training_enabled=True):
        machine_configs = self.machine_configs
        csv_reader = CSVReader(self.jobs_csv)
        jobs_configs = csv_reader.generate(self.jobs_offset, self.jobs_len)
        episode = Episode(machine_configs, jobs_configs, algorithm, "./event_file.json")
        episode.run()
        return super().start_episode(episode_id, training_enabled)

    def get_action(self, episode_id, observation_dict):
        return super().get_action(episode_id, observation_dict)

    def log_action(self, episode_id, observation_dict, action_dict):
        return super().log_action(episode_id, observation_dict, action_dict)

    def log_returns(self, episode_id, reward_dict, info_dict=None, multiagent_done_dict=None):
        return super().log_returns(episode_id, reward_dict, info_dict, multiagent_done_dict)

    def end_episode(self, episode_id, observation_dict):
        return super().end_episode(episode_id, observation_dict)