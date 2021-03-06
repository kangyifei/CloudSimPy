from core.algorithm import Algorithm
from core.cooling_equipment import CoolingEquipment
from keras.models import load_model
from typing import Dict
import numpy as np

model_path = "D:\code\CloudSimPy\core\CRAC\predict_condition_inlet_15ServerInletTempin_1out_0.15.hdf5"
class ControlAlgorithm(Algorithm):
    def __init__(self):
        self.call_num=0
        self.model=load_model(model_path)
    def __call__(self, cluster, clock, cooling_equip: CoolingEquipment = None):
        # self.call_num+=1
        # print("ControlAlgorithm",self.call_num)
        input = []
        for machine in cluster.machines:
            input.append(machine.inlet_temp)
        output = self.model.predict(np.array(input).reshape(1, 15))
        return {"inlet_temp": output.tolist()[0][0]}
