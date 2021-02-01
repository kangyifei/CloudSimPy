from core.algorithm import Algorithm
import numpy as np


class PID(Algorithm):
    def __init__(self, temp):
        self.aimingTemp = temp
        self.lastTemp = None
        self.lastError = 0
        self.errorSum = 0
        self.kp = 0.2
        self.ki = 0.1
        self.kd = 0.3

    def __call__(self, cluster, clock, cooling_equip=None):
        machinesInletTemp = []
        for machine in cluster.machines:
            machinesInletTemp.append(machine.inlet_temp)
        mean = np.array(machinesInletTemp).mean()
        errorNow = abs(self.aimingTemp - mean)
        self.errorSum += errorNow
        output = self.kp * errorNow + self.ki * self.errorSum + self.kd * (errorNow - self.lastError)
        self.lastError = errorNow
        return output

