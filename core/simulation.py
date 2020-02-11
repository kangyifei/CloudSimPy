from core.task_status_monitor import TaskStatusMonitor
from core.power_status_monitor import PowerStateMonitor


class Simulation(object):
    def __init__(self, env, cluster, task_broker, scheduler, event_file, cooling_equipment):
        self.env = env
        self.cluster = cluster
        self.task_broker = task_broker
        self.scheduler = scheduler
        self.event_file = event_file
        self.cooling_equipment = cooling_equipment
        self.monitor = []
        self.job_added_event = env.event()
        self.job_finished_event = env.event()
        if event_file is not None:
            self.monitor.append(TaskStatusMonitor(self))
        if cooling_equipment is not None:
            self.monitor.append(PowerStateMonitor(self))
            self.cooling_equipment.attach(self)
        self.cluster.attach(self)
        self.task_broker.attach(self)
        self.scheduler.attach(self)

    def run(self):
        # Starting monitor process before task_broker process
        # and scheduler process is necessary for log records integrity.
        for mon in self.monitor:
            self.env.process(mon.run())
        if self.cooling_equipment is not None:
            self.env.process(self.cooling_equipment.run())
        self.env.process(self.task_broker.run())
        self.env.process(self.scheduler.run())

    @property
    def finished(self):
        return self.task_broker.destroyed \
               and len(self.cluster.unfinished_jobs) == 0
