from enum import Enum





class MachineDoor(Enum):
    TASK_IN = 0
    TASK_OUT = 1
    NULL = 3


class Machine(object):
    def __init__(self, machine_config):
        self.id = machine_config.id
        self.cpu_capacity = machine_config.cpu_capacity
        self.memory_capacity = machine_config.memory_capacity
        self.disk_capacity = machine_config.disk_capacity
        self.cpu = machine_config.cpu
        self.memory = machine_config.memory
        self.disk = machine_config.disk
        self.cal_inlet_temp = machine_config.cal_inlet_temp
        self.cluster = None
        self.task_instances = []
        self.machine_door = MachineDoor.NULL
        self.inlet_temp = 20

    def run_task_instance(self, task_instance):
        self.cpu -= task_instance.cpu
        self.memory -= task_instance.memory
        self.disk -= task_instance.disk
        self.task_instances.append(task_instance)
        # self.inlet_temp = self.cal_inlet_temp(self.id, self.cluster)
        self.machine_door = MachineDoor.TASK_IN

    def stop_task_instance(self, task_instance):
        self.cpu += task_instance.cpu
        self.memory += task_instance.memory
        self.disk += task_instance.disk
        self.machine_door = MachineDoor.TASK_OUT
        # print("machine: one task finished")
        self.cluster.simulation.job_event.succeed(value="finished")
        self.cluster.cluster_task_finished_num+=1
        self.cluster.simulation.job_event = self.cluster.simulation.env.event()
        # print("a task finished")

    @property
    def running_task_instances(self):
        ls = []
        for task_instance in self.task_instances:
            if task_instance.started and not task_instance.finished:
                ls.append(task_instance)
        return ls

    @property
    def finished_task_instances(self):
        ls = []
        for task_instance in self.task_instances:
            if task_instance.finished:
                ls.append(task_instance)
        return ls

    def attach(self, cluster):
        self.cluster = cluster

    def accommodate(self, task):
        return self.cpu >= task.task_config.cpu and \
               self.memory >= task.task_config.memory and \
               self.disk >= task.task_config.disk

    @property
    def feature(self):
        return [self.cpu, self.memory, self.disk]

    @property
    def capacity(self):
        return [self.cpu_capacity, self.memory_capacity, self.disk_capacity]

    @property
    def state(self):
        return {
            'id': self.id,
            'inlet_temp': self.inlet_temp,
            'cpu_capacity': self.cpu_capacity,
            'memory_capacity': self.memory_capacity,
            'disk_capacity': self.disk_capacity,
            'cpu_usage': self.cpu,
            'memory_usage': self.memory,
            'disk_usage': self.disk,
            'cpu_usage_percent': 1-(self.cpu / self.cpu_capacity),
            'memory_usage_percent': 1-(self.memory / self.memory_capacity),
            'disk_usage_percent': 1-(self.disk / self.disk_capacity),
            'running_task_instances': len(self.running_task_instances),
            'finished_task_instances': len(self.finished_task_instances)
        }

    def __eq__(self, other):
        return isinstance(other, Machine) and other.id == self.id
