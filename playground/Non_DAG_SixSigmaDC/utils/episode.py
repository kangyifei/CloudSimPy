import simpy
from core.cluster import Cluster
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation
from core.cooling_equipment import CoolingEquipment
from core.config import MachineConfig, CoolingEquipmentConfig


class Episode(object):
    def __init__(self, machine_configs, task_configs, algorithm, event_file,
                 cooling_equipment):
        self.env = simpy.Environment()
        cluster = Cluster()
        cluster.add_machines(machine_configs)
        cooling_equipment = cooling_equipment
        task_broker = Broker(self.env, task_configs)

        scheduler = Scheduler(self.env, algorithm,cooling_equipment)

        self.simulation = Simulation(self.env, cluster, task_broker, scheduler, event_file, cooling_equipment)

    def run(self):
        self.simulation.run()
        self.env.run()
