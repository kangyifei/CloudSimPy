from abc import ABC


class RewardGiver(ABC):
    def __init__(self):
        self.simulation = None

    def attach(self, simulation):
        self.simulation = simulation

    def get_reward(self):
        if self.simulation is None:
            raise ValueError('Before calling method get_reward, the reward giver '
                             'must be attach to a simulation using method attach.')


class MakespanRewardGiver(RewardGiver):
    name = 'Makespan'

    def __init__(self, reward_per_timestamp):
        super().__init__()
        self.reward_per_timestamp = reward_per_timestamp

    def get_reward(self):
        super().get_reward()
        return self.reward_per_timestamp


class AverageSlowDownRewardGiver(RewardGiver):
    name = 'AS'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_tasks = cluster.unfinished_tasks
        reward = 0
        for task in unfinished_tasks:
            reward += (- 1 / task.task_config.duration)
        return reward


class AverageCompletionRewardGiver(RewardGiver):
    name = 'AC'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        return - unfinished_task_len
class TotalEnergyConsumeRewardGiver(RewardGiver):
    name='TEC'

    def get_reward(self):
        super().get_reward()
        cluster=self.simulation.cluster
        machine_states=cluster.state['machine_states']
        sum=0
        for machine_state in machine_states:
            cpu_usage_percent=machine_state['cpu_usage_percent']
            memory_usage_percent=machine_state['memory_usage_percent']
            energy_consume=cpu_usage_percent+memory_usage_percent*0.2
            sum+=(-energy_consume)
        return sum