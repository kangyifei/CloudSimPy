import json
from core.machine import Machine


class TaskStatusMonitor(object):
    def __init__(self, simulation):
        self.simulation = simulation
        self.env = simulation.env
        self.event_file = simulation.event_file
        self.events = []
        self.mean_machine_power = 0
        self.crac_power = 0
        self.total_energy_consume = 0
        self.total_crac_consume = 0
        self.time_penal=0

    def run(self):
        while not self.simulation.finished:
            temp_sum_machine_power = 0
            for machine in self.simulation.cluster.machines:
                self.total_energy_consume += self.cal_machine_power(machine)
                temp_sum_machine_power += self.cal_machine_power(machine)
            self.mean_machine_power = temp_sum_machine_power / len(self.simulation.cluster.machines)
            CRAC_power = 0
            if self.simulation.cooling_equipment is not None:
                CRAC_inlet_temp = self.simulation.cooling_equipment.state_paraslist["inlet_temp"]
                CRAC_set_temp = self.simulation.cooling_equipment.control_paramslist["set_temp"]
                if (CRAC_inlet_temp - CRAC_set_temp < 0):
                    CRAC_power = 0
                else:
                    CRAC_power = 100 * (CRAC_inlet_temp - CRAC_set_temp)
            self.crac_power = CRAC_power
            self.total_crac_consume += CRAC_power
            self.time_penal-=8
            state = {
                'timestamp': self.env.now,
                'cluster_state': self.simulation.cluster.state,
                "total_energy_consume": self.total_energy_consume,
                'total_crac_consume': self.total_crac_consume,
                'mean_machine_power': self.mean_machine_power
            }
            self.events.append(state)
            yield self.env.timeout(1)

        state = {
            'timestamp': self.env.now,
            'cluster_state': self.simulation.cluster.state
        }
        self.events.append(state)

        self.write_to_file()

    def cal_machine_power(self, machine: Machine):
        cpu_usage = machine.state["cpu_usage_percent"]
        memory_usage = machine.state['memory_usage_percent']
        if cpu_usage < 0.01:
            cpu_usage = 0
        elif cpu_usage > 0.99:
            cpu_usage = 1
        if memory_usage < 0.01:
            memory_usage = 0
        elif memory_usage > 0.99:
            memory_usage = 1
        # return 100 * cpu_usage ** 1.9 + 5 * memory_usage
        return 100*(2*cpu_usage-cpu_usage**1.4)+5*memory_usage

    def write_to_file(self):
        with open(self.event_file, 'w') as f:
            json.dump(self.events, f, indent=4)
