from core.calculate_machine_inlet_temp import CalculateMachineInletTemp
from keras.models import load_model
import numpy as np

class CpuPlusSettingTemp(CalculateMachineInletTemp):
    def __call__(self, cluster, cooling_equip):
        model_path = "../model/predict_server_inlet_1ConditionerOutletTemp+15ServerCpuUsage_15out_2.33.hdf5"
        model = load_model(model_path)
        input=[cooling_equip.get_setting_temp()]
        for machine in cluster.machines:
            input.append((1-machine.cpu/machine.cpu_capacity)*100)
        output=model.predict(np.array(input).reshape(1,16))
        output=output.reshape(15,1).tolist()
        return output

