from core.scheduler import Scheduler
from core.cooling_equipment import CoolingEquipment
from core.gym.cloudsim_env import CloudSimEnv
class Trainer(Scheduler):
    def __init__(self, simpy_env,agent,gym_env:CloudSimEnv, task_algorithm, cooling_equipment: CoolingEquipment = None):
        self.simpy_env = simpy_env
        self.gym_env=gym_env
        self.agent=agent
        self.task_algorithm = task_algorithm
        self.cooling_equipment = cooling_equipment
        self.simulation = None
        self.cluster = None
        self.destroyed = False

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster

    def task_schedule(self):
        while True:
            machine, task, = self.task_algorithm(self.cluster, self.env.now)
            print("task len:", len(self.cluster.tasks_which_has_waiting_instance))
            if machine is None or task is None:
                break
            else:
                task.start_task_instance(machine)
                print("new task started")

    def cooling_schedule(self):
        if self.cooling_equipment is not None:
            self.cooling_equipment.update_self()
            self.cooling_equipment.update_cluster()

    def job_added_schedule(self):
        self.task_schedule()
        self.cooling_schedule()

    def job_finished_schedule(self):
        self.cooling_schedule()

    def run(self):
        while not self.simulation.finished:
            print("scheduler:", "before yield")
            yield self.simulation.job_added_event | self.simulation.job_finished_event
            print("scheduler:", "after yield")
            if self.simulation.job_added_event.ok:
                ob=self.gym_env.__get_observation__(True)
                action=self.agent.choose_action(ob)

            # yield self.env.timeout(1)
        self.destroyed = True
