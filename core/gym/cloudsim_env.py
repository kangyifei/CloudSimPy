import gym
from core.gym.abstract_agent import AbstractAgent
from core.cluster import Cluster
from core.scheduler import Scheduler
from core.cooling_equipment import CoolingEquipment
from core.simulation import Simulation
from gym import spaces
from core.config import MachineConfig, JobConfig, CoolingEquipmentConfig
from typing import List, Dict
from ray.rllib.agents import ppo,dqn
d=dqn.DQNTrainer()
p=ppo.PPOTrainer(env='')
p.train()


class CloudSimEnv(gym.Env):

    def render(self, mode='human'):
        pass

    def __init__(self, machines_config_list: List[MachineConfig],
                 agent: AbstractAgent,
                 simulation: Simulation,
                 cooling_equipment_config: CoolingEquipmentConfig = None) -> None:
        super().__init__()
        self.__init_observation_action_space__(machines_config_list, cooling_equipment_config)
        # self.__init_cluster__(machines_config_list, task_config_list, cooling_equipment_config)
        self.destroyed = False
        self.agent = agent
        self.simulation = simulation
        self.cluster: Cluster = simulation.cluster
        self.cooling_equipment: CoolingEquipment = simulation.cooling_equipment
        self.task_broker = simulation.task_broker

    # def __init_cluster__(self, machines_config_list: List[MachineConfig],
    #                      task_config_list: List[JobConfig],
    #                      cooling_equipment_config: CoolingEquipmentConfig = None) -> None:
    #     # init class instance
    #     self.agent=None
    #     self.env = simpy.Environment()
    #     self.cluster = Cluster()
    #     self.cluster.add_machines(machines_config_list)
    #     self.task_broker = Broker(self.env, task_config_list)
    #     if cooling_equipment_config is not None:
    #         self.cooling_equipment = CoolingEquipment(cooling_equipment_config)
    #     self.job_added_event = self.env.event()
    #     self.job_finished_event = self.env.event()
    #     # attach all instance to self
    #     self.cluster.attach(self)
    #     self.task_broker.attach(self)
    #     # make process ready
    #     if cooling_equipment_config is not None:
    #         self.env.process(self.cooling_equipment.run())
    #     self.env.process(self.task_broker.run())

    def __init_observation_action_space__(self, machines_config_list: List[MachineConfig],
                                          cooling_equipment_config: CoolingEquipmentConfig = None) -> None:
        temp_observation_space = []
        temp_action_space = []
        # add machines params
        for machine_config in machines_config_list:
            temp_observation_space.append(spaces.Box(low=0, high=machine_config.cpu_capacity, shape=(1,)))
            temp_observation_space.append(spaces.Box(low=0, high=machine_config.memory_capacity, shape=(1,)))
            temp_observation_space.append(spaces.Box(low=0, high=machine_config.disk_capacity, shape=(1,)))
        temp_action_space.append(spaces.Discrete(len(machines_config_list)))
        # add cooling equipments params
        if cooling_equipment_config is not None:
            state_paramslist = cooling_equipment_config.state_paramslist
            for state_param_key in state_paramslist:
                temp_observation_space.append(spaces.Box(low=state_paramslist[state_param_key]["low"],
                                                         high=state_paramslist[state_param_key]["high"],
                                                         shape=(1,)))
            control_paramslist = cooling_equipment_config.control_paramslist
            for control_param_key in control_paramslist:
                temp_action_space.append(spaces.Box(low=control_paramslist[control_param_key]["low"],
                                                    high=control_paramslist[control_param_key]["high"],
                                                    shape=(1,)))
        # add coming task params including cpu,mem,disk
        maxmum_cpu_capacity = max([machine_config.cpu_capacity for machine_config in machines_config_list])
        maxmum_mem_capacity = max([machine_config.memory_capacity for machine_config in machines_config_list])
        maxmum_disk_capacity = max([machine_config.disk_capacity for machine_config in machines_config_list])
        temp_observation_space.append(spaces.Box(low=0, high=maxmum_cpu_capacity, shape=(1,)))
        temp_observation_space.append(spaces.Box(low=0, high=maxmum_mem_capacity, shape=(1,)))
        temp_observation_space.append(spaces.Box(low=0, high=maxmum_disk_capacity, shape=(1,)))
        # transform to real space
        self.observation_space = spaces.Tuple(tuple(temp_observation_space))
        self.action_space = spaces.Tuple(tuple(temp_action_space))

    def __get_next_task__(self):
        return self.cluster.tasks_which_has_waiting_instance[0]

    # action:{"machine":0,"cooling_equipment":List[]}
    def step(self, action: Dict):
        if action["machine"] is not None:
            scheduled_machine = self.cluster.machines[action["machine"]]
            self.next_task.start_task_instance(scheduled_machine)
            print("new task started at ", scheduled_machine)
            action_cooling_params_list = action["cooling_equipment"]
            i = 0
            for paramskey in self.cooling_equipment.control_paramslist:
                self.cooling_equipment.control_paramslist[paramskey] = action_cooling_params_list[i]
                i += 1
            self.cooling_equipment.update_self()
            self.cooling_equipment.update_cluster()
            return self.__get_observation__(False), self.__get__reward__(), self.finished, None
        else:
            action_cooling_params_list = action["cooling_equipment"],
            i = 0
            for paramskey in self.cooling_equipment.control_paramslist:
                self.cooling_equipment.control_paramslist[paramskey] = action_cooling_params_list[i]
                i += 1
            self.cooling_equipment.update_self()
            self.cooling_equipment.update_cluster()
        return self.__get_observation__(False), self.__get__reward__(), self.finished, None

    def __get_observation__(self, new_job=True):
        observation = []
        for machine in self.cluster.machines:
            observation.append(machine.cpu)
            observation.append(machine.memory)
            observation.append(machine.disk)
        for paramskey in self.cooling_equipment.state_paraslist:
            observation.append(self.cooling_equipment.state_paraslist[paramskey])

        if new_job:
            self.next_task = self.__get_next_task__()
            observation.append(self.next_task.task_config.cpu)
            observation.append(self.next_task.task_config.memory)
            observation.append(self.next_task.task_config.disk)
        else:
            observation.append(0)
            observation.append(0)
            observation.append(0)
        return observation

    def __get__reward__(self):
        return 1

    def job_added_schedule(self):
        ob = self.__get_observation__(new_job=True)
        reward = self.__get__reward__()
        while self.cluster.tasks_which_has_waiting_instance:
            # choose action
            action = self.agent.choose_action(ob)
            # do task schedule action
            scheduled_machine = self.cluster.machines[action[0]]
            self.next_task.start_task_instance(scheduled_machine)
            print("new task started at ", scheduled_machine)
            # do cooling schedule action
            action_cooling_params_list = action[1:]
            i = 0
            for paramskey in self.cooling_equipment.control_paramslist:
                self.cooling_equipment.control_paramslist[paramskey] = action_cooling_params_list[i]
                i += 1
            # agent learn
            self.agent.learn(ob, action, reward)
            # update cooling equipment
            self.cooling_equipment.update_self()
            # get new ob,reward
            ob = self.__get_observation__(new_job=True)
            reward = self.__get__reward__()

    def job_finished_schedule(self):
        ob = self.__get_observation__(new_job=False)
        reward = self.__get__reward__()
        while self.cluster.tasks_which_has_waiting_instance:
            # choose action
            action = self.agent.choose_action(ob)
            # do cooling schedule action
            action_cooling_params_list = action[1:]
            i = 0
            for paramskey in self.cooling_equipment.control_paramslist:
                self.cooling_equipment.control_paramslist[paramskey] = action_cooling_params_list[i]
                i += 1
            # agent learn
            self.agent.learn(ob, action, reward)
            # update cooling equipment
            self.cooling_equipment.update_self()
            self.cooling_equipment.update_cluster()
            # get new ob,reward
            ob = self.__get_observation__(new_job=False)
            reward = self.__get__reward__()

    def run(self):
        while not self.simulation.finished:
            yield self.simulation.job_added_event | self.simulation.job_finished_event
            if self.simulation.job_added_event.ok:
                print("job added")
                self.job_added_schedule()
            else:
                print("job finished")
                self.job_finished_schedule()
        self.destroyed = True

    @property
    def finished(self):
        return self.task_broker.destroyed \
               and len(self.cluster.unfinished_jobs) == 0

    def reset(self):
        pass
