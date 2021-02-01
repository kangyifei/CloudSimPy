from core.gym.cloudsim_env import CloudSimEnv
from core.config import MachineConfig,CoolingEquipmentConfig
import numpy as np
import json
import simpy
from core.cluster import Cluster
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation
machine_ls=[]
cooling_ls=[]
for _ in range(3):
    machine_ls.append(MachineConfig(64, 1, 1))
c=CloudSimEnv(machine_ls,CoolingEquipmentConfig("../CRAC.json"))
ob=c.observation_space.sample()
ac=c.action_space.sample()
obn=np.array(ob)
print(ob)