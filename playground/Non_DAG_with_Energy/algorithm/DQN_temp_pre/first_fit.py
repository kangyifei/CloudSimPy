from core.algorithm import Algorithm
from typing import Optional, Any
import numpy as np
from keras.models import load_model


class FirstFitTaskalgorithm(Algorithm):

    def __init__(self):
        self.temp_model = load_model(
            "D:/code/CloudSimPy/playground/Non_DAG_with_Energy/algorithm/DQN_temp_pre/predict_temp_usingcpu_ts36_ph12_0.19.hdf5")
        self.cpu_model = load_model(
            "D:/code/CloudSimPy/playground/Non_DAG_with_Energy/algorithm/DQN_temp_pre/predict_cpu_ts36_ph12_0.19.hdf5")

    def __call__(self, cluster, clock, cooling_equipment=None) -> (Any, Any, Any):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        candidate_task = None
        candidate_machine = None
        final_i = 0
        for i in range(len(machines)):
            machine = machines[i]
            for task in tasks:
                if machine.accommodate(task):
                    candidate_machine = machine
                    candidate_task = task
                    final_i = i
                    break
            if candidate_machine is not None:
                break
        if (candidate_machine is not None):
            temp = self.get_pre_temp(cluster, final_i)
            cluster.machines[i].inlet_temp = temp
        return candidate_machine, candidate_task, None

    def get_pre_temp(self, cluster, machine_id):
        if len(cluster.monitor.events) < 40:
            return 26
        events = cluster.monitor.events[-36:]
        inlettemp_list = []
        cpu_list = []
        for e in events:
            machine_inlettemp = e['cluster_state']['machine_states'][machine_id]['inlet_temp']
            machine_cpu = e['cluster_state']['machine_states'][machine_id]['cpu_usage_percent']
            inlettemp_list.append(machine_inlettemp)
            cpu_list.append(machine_cpu)
        cpu_list = np.array(cpu_list)
        cpu_list = cpu_list.reshape(-1, cpu_list.shape[0])
        pred_cpu = self.cpu_model.predict(cpu_list)
        pred_cpu = pred_cpu.tolist()[0][0]
        inlettemp_list.append(pred_cpu)
        inlettemp_list = np.array(inlettemp_list)
        inlettemp_list = inlettemp_list.reshape(-1, inlettemp_list.shape[0])
        pred_temp = self.temp_model.predict(inlettemp_list)
        return pred_temp.tolist()[0][0]
