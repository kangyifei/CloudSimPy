from keras.models import load_model
import numpy as np

class CoolingEquipmentConfig(object):
    def __init__(self, min_temp=None,max_temp=None):
        # self.cal_machine_inlet_temp_method=cal_machine_inlet_temp_method
        # self.cal_self_inlet_temp_method=cal_self_inlet_temp_method
        self.min_temp=min_temp
        self.max_temp=max_temp


class CoolingEquipment(object):

    def __init__(self,coolingconfig:CoolingEquipmentConfig):
        self.setting_temp = 22
        self.inlet_temp = 22
        self.min_temp=0 if coolingconfig.min_temp is None else coolingconfig.min_temp
        self.max_temp=30 if coolingconfig.max_temp is None else coolingconfig.max_temp
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
    def cal_machines_temp(self):
        model_path = "./predict_server_inlet_1ConditionerOutletTemp+15ServerCpuUsage_15out_2.33.hdf5"
        model = load_model(model_path)
        input=[self.get_setting_temp()]
        for machine in self.simulation.cluster.machines:
            input.append((1-machine.cpu/machine.cpu_capacity)*100)
        output=model.predict(np.array(input).reshape(1,16))
        output=output.reshape(15,1).tolist()
        return output
    def cal_self_temp(self):
        model_path = "./predict_condition_inlet_15ServerInletTempin_1out_0.15.hdf5"
        model = load_model(model_path)
        input=[]
        for machine in self.simulation.cluster.machines:
            input.append(machine.inlet_temp)
        output=model.predict(np.array(input).reshape(1,15))
        return output[0]

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
