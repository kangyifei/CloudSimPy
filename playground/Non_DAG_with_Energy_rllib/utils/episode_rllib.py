import simpy
from core.cluster import Cluster
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation
from ray.rllib.env import ExternalEnv
from gym import spaces
from core.config import MachineConfig, CoolingEquipmentConfig
from typing import List
import numpy as np
from playground.Non_DAG_with_Energy_rllib.utils.csv_reader import CSVReader
from playground.Non_DAG_with_Energy_rllib.algorithm.DQN_rllib.DQN_algorithm import DQNAlgorithm

machines_number=5
jobs_len=2
class Episode(ExternalEnv):
    def __init__(self, config):
        jobs_csv = "C:\\code\\CloudSimPy\\CloudSimPy\\playground\\Non_DAG_with_Energy_rllib\\jobs.csv"
        machine_configs = [MachineConfig(64, 1, 1) for _ in range(machines_number)]
        csv_reader = CSVReader(jobs_csv)
        task_configs = csv_reader.generate(10, jobs_len)
        algorithm = DQNAlgorithm(machine_configs)
        event_file=None
        self.__init_observation_action_space__(machine_configs, None)
        super().__init__(self.action_space, self.observation_space)
        self.algorithm = algorithm
        self.env = simpy.Environment()
        cluster = Cluster()
        cluster.add_machines(machine_configs)

        task_broker = Broker(self.env, task_configs)

        scheduler = Scheduler(self.env, self.algorithm)
        self.simulation = Simulation(self.env, cluster, task_broker, scheduler, event_file)

    def __init_observation_action_space__(self, machines_config_list: List[MachineConfig],
                                          cooling_equipment_config: CoolingEquipmentConfig = None) -> None:
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

    def run(self):
        self.eid = self.start_episode()
        self.algorithm.attach(self)
        self.simulation.run()
        self.env.run()
        observation = []
        machines = self.simulation.cluster.machines
        for machine in machines:
            observation.append(machine.cpu)
            observation.append(machine.memory)
        observation.append(0)
        observation.append(0)
        observation = np.array(observation)
        self.end_episode(self.eid, observation)
