import numpy as np
from core.algorithm import Algorithm
from keras.models import load_model

class Tetris(Algorithm):
    def __init__(self):
        self.temp_model = load_model(
            "D:/code/CloudSimPy/playground/Non_DAG_with_Energy/algorithm/DQN_temp_pre/predict_temp_usingcpu_ts36_ph12_0.19.hdf5")
        self.cpu_model = load_model(
            "D:/code/CloudSimPy/playground/Non_DAG_with_Energy/algorithm/DQN_temp_pre/predict_cpu_ts36_ph12_0.19.hdf5")

    @staticmethod
    def calculate_alignment(valid_pairs):
        machine_features = []
        task_features = []
        for index, pair in enumerate(valid_pairs):
            machine = pair[0]
            task = pair[1]
            machine_features.append(machine.feature[:2])
            task_features.append([task.task_config.cpu, task.task_config.memory])
        return np.argmax(np.sum(np.array(machine_features) * np.array(task_features), axis=1), axis=0)

    def __call__(self, cluster, clock, cooling_equipment=None):
        machines = cluster.machines
        tasks = cluster.tasks_which_has_waiting_instance
        valid_pairs = []
        pairs2machine_index = []
        for i in range(len(machines)):
            machine=machines[i]
            for task in tasks:
                if machine.accommodate(task):
                    valid_pairs.append((machine, task))
                    pairs2machine_index.append(i)
        if len(valid_pairs) == 0:
            return None, None, None
        pair_index = Tetris.calculate_alignment(valid_pairs)
        pair = valid_pairs[pair_index]
        machine_index = pairs2machine_index[pair_index]
        temp = self.get_pre_temp(cluster, machine_index)
        pair[0].inlet_temp = temp
        return pair[0], pair[1], None

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
