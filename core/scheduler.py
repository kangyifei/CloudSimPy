class Scheduler(object):
    def __init__(self, env, task_algorithm, cooling_algorithm=None):
        self.env = env
        self.task_algorithm = task_algorithm
        self.cooling_algorithm = None if cooling_algorithm is None else cooling_algorithm
        self.simulation = None
        self.cluster = None
        self.destroyed = False
        self.valid_pairs = {}

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def task_schedule(self):
        while True:
            machine, task, = self.task_algorithm(self.cluster, self.env.now)
            if machine is None or task is None:
                break
            else:
                task.start_task_instance(machine)

    def cooling_schedule(self):
        if self.cooling_algorithm is not None:
            setting_temp = self.cooling_algorithm(self.cluster, self.env.now,self.simulation.cooling_equipment)
            self.cluster.cooling_equipment.set_temp(setting_temp)

    def run(self):
        while not self.simulation.finished:
            yield self.simulation.job_added_event | self.simulation.job_finished_event
            self.task_schedule()
            self.cooling_schedule()
            # yield self.env.timeout(1)
        self.destroyed = True
