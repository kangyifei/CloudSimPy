import numpy as np
import sys

sys.path.append('..')
from core.config import MachineConfig
import ray
from ray.rllib.agents.dqn import DQNTrainer

np.random.seed(41)
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 2
n_iter = 2000
jobs_csv = './jobs.csv'
machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
print("-----------------------------------------drl------------------------------------------")
total_energy_consume_list = []
total_called_num_list = []
total_reward_of_ep_list = []
average_completion_list = []
average_waitting_time_list = []
ray.init(address='192.168.1.100:6379')
# failed-1
# config=ray.rllib.agents.dqn.DEFAULT_CONFIG.copy()
# config["num_gpus"]=1
# config["num_workers"]=1
# from ray.tune.registry import register_env
# from ray.tune.logger import pretty_print
# register_env("cloudsim",lambda config:Episode(config))
# trainer=DQNTrainer(config=config,env="cloudsim")

# for i in range(100):
#     res=trainer.train()
#     print(pretty_print(res))

# failed-2
# tune.run(DQNTrainer,config=config)

import gym
from gym import spaces
import random


class CloudSimStub(gym.Env):
    def __init__(self, config):
        self.__init_observation_action_space__(machine_configs, None)

    def __init_observation_action_space__(self,
                                          machines_config_list,
                                          cooling_equipment_config=None) -> None:
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
        box_low_bound.append(0)
        box_high_bound.append(maxmum_cpu_capacity)
        box_low_bound.append(0)
        box_high_bound.append(maxmum_mem_capacity)
        box_low_bound = np.array(box_low_bound)
        box_high_bound = np.array(box_high_bound)
        self.observation_space = spaces.Box(box_low_bound, box_high_bound)
        self.action_space = spaces.Discrete(len(machines_config_list) + 1)
        print(self.observation_space.shape)
        print(self.action_space.shape)
        self.steps = 0

    def reset(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        done = False
        return self.observation_space.sample(), \
               float(random.random), \
               done, \
               {}


from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

register_env("cloudsim", lambda config: CloudSimStub(config))
SERVER_ADDRESS = "0.0.0.0"
PORT = 6521
trainer_config = {
    "input": (
        lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, PORT)),
    # Use a single worker process (w/ SyncSampler) to run the server.
    "num_workers": 0,
    "dueling": True,
    "double_q": True,
    # Disable OPE, since the rollouts are coming from online clients.
    "input_evaluation": [],
    "framework": "torch",
}
checkpoint_path = "./dqn_rllib_savepoint.out"
trainer = DQNTrainer(config=trainer_config, env="cloudsim")
while True:
    print(pretty_print(trainer.train()))
    checkpoint = trainer.save()
    print("Last checkpoint", checkpoint)
    with open(checkpoint_path, "w") as f:
        f.write(checkpoint)
