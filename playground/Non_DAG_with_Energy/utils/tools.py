import time
import numpy as np


def average_completion(exp):
    completion_time = 0
    number_task = 0
    for job in exp.simulation.cluster.jobs:
        for task in job.tasks:
            number_task += 1
            completion_time += (task.finished_timestamp - task.started_timestamp)
    return completion_time / number_task


def average_waiting_time(exp):
    waiting_time = 0
    number_task = 0
    for job in exp.simulation.cluster.jobs:
        for task in job.tasks:
            number_task += 1
            waiting_time += (job.job_config.submit_time - task.started_timestamp)
    return waiting_time / number_task


def average_slowdown(exp):
    slowdown = 0
    number_task = 0
    for job in exp.simulation.cluster.jobs:
        for task in job.tasks:
            number_task += 1
            slowdown += (task.finished_timestamp - task.started_timestamp) / task.task_config.duration
    return slowdown / number_task


def total_energy_consume(exp):
    for machine in exp.simulation.cluster.machines:
        state = machine.state
        cpu = state['cpu_usage_percent']
        memory = state['memory_usage_percent']


def multiprocessing_run(episode, trajectories, makespans, average_completions, average_slowdowns):
    np.random.seed(int(time.time()))
    episode.run()
    trajectories.append(episode.simulation.scheduler.algorithm.current_trajectory)
    makespans.append(episode.simulation.env.now)
    # print(episode.simulation.env.now)
    average_completions.append(average_completion(episode))
    average_slowdowns.append(average_slowdown(episode))
