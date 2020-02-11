class CoolingEquipment(object):

    def __init__(self, env, cooling_alogrithm, cal_machine_inlet_temp_method, cal_self_inlet_temp_method):
        self.cooling_algorithm = cooling_alogrithm
        self.cal_machines_temp = cal_machine_inlet_temp_method
        self.cal_self_temp = cal_self_inlet_temp_method
        self.env = env
        self.setting_temp = 22
        self.inlet_temp = 22
        self.machines = None
        self.simulation = None
        self.cluster = None

    def attach(self, simulation):
        self.simulation = simulation
        self.cluster = simulation.cluster
        self.machines = self.cluster.machines

    def set_temp(self, temp):
        self.setting_temp = temp

    def get_setting_temp(self):
        return self.setting_temp

    def __cal_machines_inlet_temp(self):
        machines_temp = self.cal_machines_temp()
        for i in range(len(self.machines)):
            self.machines[i].inlet_temp = machines_temp[i]

    def run(self):
        while not self.simulation.finished:
            yield self.simulation.job_added_event | self.simulation.job_finished_event
            self.inlet_temp = self.cal_self_temp()
            self.__cal_machines_inlet_temp()

    @property
    def state(self):
        return {
            "setting_temp": self.setting_temp,
            "inlet_temp": self.inlet_temp
        }
