import numpy as np
from keras.models import Model
from keras.layers import Dense, Reshape, Input
from keras.layers import LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint
import csv
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
file_path = "./singlerack_all.csv"
server_list = []
conditioner_outlet_temp = []
conditioner_inlet_temp = []

timestep = 36
predict_horizon = 12


class Server(object):
    def __init__(self, i):
        self.id = i
        self.inlet_temp = []
        self.outlet_temp = []
        self.cpu = []
        self.memory = []


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)


def process_data():
    for i in range(15):
        server_list.append(Server(i))
    with open(file_path, "r", encoding='utf-8') as datacsv:
        csvr = csv.reader(datacsv)
        for row in csvr:
            i = 1
            for server in server_list:
                server.outlet_temp.append(float(row[i]))
                i = i + 1
            for server in server_list:
                server.inlet_temp.append(float(row[46 - i]))
                i = i + 1
            conditioner_outlet_temp.append(float(row[i]))
            i = i + 1
            conditioner_inlet_temp.append(float(row[i]))
            i = i + 6
            for server in server_list:
                # if(server.id<10):
                #     server.cpu.append(float(row[i])/10)
                # else:
                #     server.cpu.append(float(row[i]))
                server.cpu.append(float(row[i]) / 100)
                i = i + 6
def train_model_combined_cpu_temp():
    temp_train_x = []
    temp_train_y = []
    for server in server_list:
        temp = np.array(server.inlet_temp)
        data = strided_app(temp, timestep + predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        y = data[:, -1]
        if isinstance(temp_train_x, list):
            temp_train_x = x
            temp_train_y = y
        else:
            temp_train_x = np.concatenate((temp_train_x, x), axis=0)
            temp_train_y = np.concatenate((temp_train_y, y), axis=0)
    cpu_x = []
    for server in server_list:
        cpu = np.array(server.cpu)
        data = strided_app(cpu, timestep + predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        if isinstance(cpu_x, list):
            cpu_x = x
        else:
            cpu_x = np.concatenate((cpu_x, x), axis=0)
    temp_train_x=np.expand_dims(temp_train_x,2)
    cpu_x=np.expand_dims(cpu_x,2)
    mix_x = np.concatenate((temp_train_x, cpu_x), axis=2)
    input = Input(shape=(timestep,2))
    # re = Reshape(target_shape=(timestep * 2, 1))(input)
    lstm = LSTM(120, return_sequences=True)(input)
    lstm1 = LSTM(120)(lstm)
    flatten = lstm1
    dense = Dense(10, activation='relu')(flatten)
    output = Dense(1)(dense)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    checkpoint = ModelCheckpoint(
        "./feature_2_model/predict_temp_combined_cpu+temp_ts" + str(timestep) + "_ph" + str(
            predict_horizon) + "_{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002, patience=3, verbose=1)
    callbacks_list = [checkpoint, early_stopping]
    history = model.fit(mix_x, temp_train_y, epochs=10000,
                        batch_size=128,
                        validation_split=0.02, verbose=1, callbacks=callbacks_list)

if __name__=="__main__":
    process_data()
    train_model_combined_cpu_temp()