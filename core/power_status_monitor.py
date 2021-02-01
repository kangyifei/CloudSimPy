import json


class PowerStateMonitor(object):
    def __init__(self, simulation):
        self.simulation = simulation
        self.env = simulation.env
        self.event_file = simulation.event_file + "_power"
        self.events = []

    def __cal_machine_power(self):
        machines = self.simulation.cluster.machines
        sum = 0
        for machine in machines:
            power = 100 * machine.state['cpu_usage_percent'] + 2 * machine.state['memory_usage_percent']

            sum += power
        return sum

    def __cal_cooling_equipment_power(self):
        cooling_equipment = self.simulation.cluster.cooling_equipment
        if ((cooling_equipment.state['inlet_temp'] - cooling_equipment.state['setting_temp']) < 0):
            power = 0
        else:
            power = 100 * (cooling_equipment.state['inlet_temp'] - cooling_equipment.state['setting_temp'])
        return power

    def run(self):
        machine_power_sum = 0
        cooling_power_sum = 0
        while not self.simulation.finished:
            machine_power = round(self.__cal_machine_power(), 2)
            cooling_power = round(self.__cal_cooling_equipment_power(), 2)
            machine_power_sum += machine_power
            cooling_power_sum += cooling_power
            state = {
                'timestamp': self.env.now,
                'machine_power': machine_power,
                'cooling_power': cooling_power
            }
            self.events.append(state)
            yield self.env.timeout(1)

        state = {
            'timestamp': self.env.now,
            'machine_power_sum': machine_power_sum,
            'cooling_power_sum': cooling_power_sum
        }
        self.events.append(state)

        self.__write_to_file()

    def __write_to_file(self):
        with open(self.event_file, 'w') as f:
            json.dump(self.events, f, indent=4)
