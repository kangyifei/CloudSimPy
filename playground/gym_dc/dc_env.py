import gym
from gym import spaces
import simpy
from core.cluster import Cluster
from core.cooling_equipment import CoolingEquipment,CoolingEquipmentConfig
from core.broker import Broker
from core.machine import MachineConfig
from playground.Non_DAG.utils.csv_reader import CSVReader
from .adapter.simulation import Simulation
from stable_baselines.common.env_checker import check_env
from stable_baselines.a2c import A2C
from playground.Non_DAG_with_Energy.algorithm.Cal_Machine_inlet.CpuPlusSettingTemp import CpuPlusSettingTemp



class DataCenterEnv(gym.Env):
    def __init__(self, machine_configs, task_configs,cooling_config,  event_file, reward_giver,
                 temp_threshold):
        self.env = simpy.Environment()
        self.machine_configs = machine_configs
        self.task_configs = task_configs
        self.event_file = event_file
        self.cooling_config=cooling_config
        cluster = Cluster()
        cluster.add_machines(self.machine_configs)
        task_broker = Broker(self.env, self.task_configs)
        cooling=CoolingEquipment(self.cooling_config)
        self.simulation = Simulation(self.env, cluster,task_broker, self.event_file,cooling)
        self.reward_giver = reward_giver
        self.temp_threshold = temp_threshold
        self.taskpointer = 0
        # host_num,cooling_set_temp
        self.action_space = spaces.Tuple((spaces.Discrete(len(self.simulation.cluster.machines) + 1),
                                          spaces.Box(low=self.simulation.cooling_equipment.min_temp,
                                                     high=self.simulation.cooling_equipment.max_temp, shape=1)))
        # cpu_usage,mem_usage,machine_inlet_temp,
        # cooling_inlet_temp,
        # next_task_instance_cpu,next_task_instance_mem
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=self.simulation.cluster.machines[0].state[
            'cpu_capacity'], shape=len(self.simulation.cluster.machines)),
                                               spaces.Box(low=0, high=self.simulation.cluster.machines[0].state[
                                                   'memory_capacity'], shape=len(self.simulation.cluster.machines), ),
                                               spaces.Box(low=0, high=30, shape=len(self.simulation.cluster.machines)),

                                               spaces.Box(low=0, high=30, shape=1),

                                               spaces.Box(low=0, high=self.simulation.cluster.machines[0].state[
                                                   'cpu_capacity'], shape=1),
                                               spaces.Box(low=0, high=self.simulation.cluster.machines[0].state[
                                                   'memory_capacity'], shape=1)
                                               ))
        self.simulation.run()
        self.env.run()

    def step(self, action: tuple):
        tasks = self.simulation.cluster.tasks_which_has_waiting_instance
        if not self.simulation.cluster.machines[action[0]].accommodate(tasks[0]):
            done = True
            reward = 0
            return (), reward, done, {}
        if (action[0] != 0):
            tasks[0].start_task_instance(self.simulation.cluster.machines[action[0]])
        self.simulation.cooling_equipment.set_temp(action[1])
        if self.simulation.finished:
            done = True
            reward = self.reward_giver(self.simulation)
            return (), reward, done, {}
        reward = self.reward_giver(self.simulation)
        cpu_usage = []
        mem_usage = []
        for machine in self.simulation.cluster.machines:
            cpu_usage.append(machine.state['cpu'])
            mem_usage.append(machine.state['memory_capacity'])
        cool_inlet_temp = self.simulation.cooling_equipment.state['inlet_temp']
        # TODO:now is FCFS,maybe should use a priority head to avoid repeatly sche first task
        tasks = self.simulation.cluster.tasks_which_has_waiting_instance
        next_task_instance_cpu = tasks[0]
        next_task_instance_mem = tasks[0]
        obs = (cpu_usage, mem_usage, cool_inlet_temp, next_task_instance_cpu, next_task_instance_mem)
        return obs, reward, False, {}

    def reset(self):
        del self.simulation
        del self.env
        import gc
        gc.collect()
        self.env = simpy.Environment()
        cluster = Cluster()
        cluster.add_machines(self.machine_configs)
        task_broker = Broker(self.env, self.task_configs)
        cooling=CoolingEquipment(self.cooling_config)
        self.simulation = Simulation(self.env, cluster,task_broker, self.event_file,cooling)
        self.simulation.run()
        self.env.run()

if __name__ =='__main__':
    def reware_giver():
        return  1
    jobs_csv = '../jobs_files/jobs.csv'
    jobs_len = 10
    machines_number=15
    machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
    csv_reader = CSVReader(jobs_csv)
    jobs_configs = csv_reader.generate(0, jobs_len)
    cooling_config=CoolingEquipmentConfig(0,30)
    env = DataCenterEnv(machine_configs,jobs_configs,cooling_config,None,reware_giver(),26)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
