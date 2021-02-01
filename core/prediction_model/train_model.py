import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Flatten, Reshape, Input
from keras.layers import LSTM
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint, EarlyStopping
import json
import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
file_path = "D:\\邢老师数据集\\singlerack_all.csv"
server_list = []
conditioner_outlet_temp = []
conditioner_inlet_temp = []


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
                server.cpu.append(float(row[i]) /100)
                i = i + 6


def format_dataset():
    predict_server_inlet_model_x = []
    predict_server_inlet_model_y = []
    predict_conditioner_inlet_model_x = []
    predict_conditioner_inlet_model_y = conditioner_inlet_temp
    for i in range(len(conditioner_outlet_temp)):
        x_row = []
        y_row = []
        x_row.append(conditioner_outlet_temp[i])
        for server in server_list:
            x_row.append(server.cpu[i])
            y_row.append(server.inlet_temp[i])
        predict_server_inlet_model_x.append(x_row)
        predict_server_inlet_model_y.append(y_row)
        predict_conditioner_inlet_model_x.append(y_row)
    return np.array(predict_server_inlet_model_x), np.array(predict_server_inlet_model_y), np.array(
        predict_conditioner_inlet_model_x), np.array(predict_conditioner_inlet_model_y)


def train_server_model(predict_server_inlet_model_x, predict_server_inlet_model_y):
    server_model = Sequential()
    # server_model.add(LSTM(10, activation="relu", input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True,
    #                       kernel_initializer=RandomNormal()))
    # server_model.add(Flatten())
    server_model.add(Dense(16, activation="relu"))
    server_model.add(Dense(100, activation="relu"))
    # server_model.add(Dense(500, activation="relu"))
    # server_model.add(Dense(100, activation="relu"))
    server_model.add(Dense(15))
    server_model.compile(loss='mean_absolute_error', optimizer='Adadelta')
    checkpoint1 = ModelCheckpoint(
        "./model/predict_server_inlet_1ConditionerOutletTemp+15ServerCpuUsage_15out_{val_loss:.2f}.hdf5",
        monitor='val_loss', verbose=1, save_best_only=True,
        mode='min')
    callbacks_list1 = [checkpoint1]
    history1 = server_model.fit(predict_server_inlet_model_x, predict_server_inlet_model_y, epochs=10000,
                                batch_size=256,
                                validation_split=0.2, verbose=2, callbacks=callbacks_list1)


def train_cpu_usage_model():
    timestep = 60
    predict_horizon=60
    train_x = []
    train_y = []
    for server in server_list:
        cpu = np.array(server.cpu)
        data = strided_app(cpu, timestep + +predict_horizon+1, 1)
        x = data[:, :-1-predict_horizon]
        y = data[:, -1]
        if isinstance(train_x,list):
            train_x = x
            train_y = y
        else:
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)
    input=Input(shape=(timestep,))
    re=Reshape(target_shape=(timestep, 1))(input)
    lstm=LSTM(120, return_sequences=True)(re)
    lstm1 = LSTM(120)(lstm)
    flatten=lstm1
    dense=Dense(10,activation='relu')(flatten)
    output=Dense(1)(dense)
    model=Model(inputs=[input],outputs=[output])
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    checkpoint = ModelCheckpoint("./model/predict_cpu_usage_ts60_ph60_{val_loss:.2f}.hdf5",
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002, patience=3, verbose=1)
    callbacks_list = [checkpoint, early_stopping]
    history = model.fit(train_x, train_y, epochs=10000,
                        batch_size=128,
                        validation_split=0.02, verbose=1, callbacks=callbacks_list)


def train_conditioner_model(predict_conditioner_inlet_model_x, predict_conditioner_inlet_model_y):
    conditioner_model = Sequential()
    conditioner_model.add(Dense(15, activation="relu"))
    conditioner_model.add(Dense(100, activation="relu"))
    conditioner_model.add(Dense(1))
    conditioner_model.compile(loss='mse', optimizer='Adadelta')
    checkpoint2 = ModelCheckpoint("./model/predict_condition_inlet_15ServerInletTempin_1out_{val_loss:.2f}.hdf5",
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002, patience=10, verbose=1)
    callbacks_list2 = [checkpoint2]
    history2 = conditioner_model.fit(predict_conditioner_inlet_model_x, predict_conditioner_inlet_model_y, epochs=10000,
                                     batch_size=256,
                                     validation_split=0.2, verbose=2, callbacks=callbacks_list2)


if __name__ == "__main__":
    process_data()
    # predict_server_inlet_model_x,\
    # predict_server_inlet_model_y, \
    # predict_conditioner_inlet_model_x,\
    # predict_conditioner_inlet_model_y = format_dataset()
    #
    # train_server_model(predict_server_inlet_model_x, predict_server_inlet_model_y)

    # train_conditioner_model(predict_conditioner_inlet_model_x,predict_conditioner_inlet_model_y)
    # res_y=server_model.predict(predict_server_inlet_model_x)
    # while(True):
    #     pass

    train_cpu_usage_model()
