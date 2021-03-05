from core.algorithm import Algorithm
from typing import Optional,Any

class FirstFitTaskalgorithm(Algorithm):
    def __call__(self, cluster, clock,cooling_equipment=None)->(Any,Any,Any):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        candidate_task = None
        candidate_machine = None

        for machine in machines:
            for task in tasks:
                if machine.accommodate(task):
                    candidate_machine = machine
                    candidate_task = task
                    break
        return candidate_machine, candidate_task,None