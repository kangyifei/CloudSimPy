import simpy
from core.cluster import Cluster
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation


class Episode(object):
    broker_cls = Broker

    def __init__(self, machine_configs, task_configs, task_algorithm, cooling_equipment=None, cooling_algorithm=None,
                 event_file=None):
        self.env = simpy.Environment()
        cluster = Cluster()
        cluster.add_machines(machine_configs)
        task_broker = Episode.broker_cls(self.env, task_configs)

        scheduler = Scheduler(self.env, task_algorithm, cooling_algorithm)

        self.simulation = Simulation(self.env, cluster, task_broker, scheduler, event_file)

    def run(self):
        self.simulation.run()
        self.env.run()
