import numpy as np
# import keras
# from keras.models import Sequential, Model
# from keras.layers import Dense, Flatten, Reshape, Input
# from keras.layers import LSTM
# from keras.initializers import RandomNormal
# from keras.callbacks import ModelCheckpoint, EarlyStopping
import json
import csv
import os
import torch
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
file_path = "./singlerack_all.csv"
server_list = []
conditioner_outlet_temp = []
conditioner_inlet_temp = []

timestep = 36
predict_horizon = 12


class SingleLSTM(nn.Module):
    def __init__(self):
        super(SingleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=120, num_layers=1)
        self.l1 = nn.Sequential(
            nn.Linear(120, 10),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x, hidden = self.lstm(x)
        x = x[35, :, :]
        x = self.l1(x)
        x = self.l2(x)
        return x


class SingleLSTMQuantization(nn.Module):
    def __init__(self):
        super(SingleLSTMQuantization, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.lstm = nn.LSTM(input_size=1, hidden_size=120, num_layers=1)
        self.l1 = nn.Linear(120, 10)
        self.l1Relu = nn.ReLU
        self.l2 = nn.Linear(10, 1)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x, (hn,cn) = self.lstm(x)
        x = x[35, :, :]
        x = self.l1(x)
        x = self.l1Relu(x)
        x = self.l2(x)
        x = self.dequant(x)
        return x


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


def train_model_without_cpu():
    train_x = []
    train_y = []
    for server in server_list:
        inlet_temp = np.array(server.inlet_temp)
        data = strided_app(inlet_temp, timestep + +predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        y = data[:, -1]
        if isinstance(train_x, list):
            train_x = x
            train_y = y
        else:
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)
    print(train_x.shape)
    train_x = train_x.transpose()
    print(train_x.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    last_loss = 0
    for _ in range(100):
        ep_loss = 0
        for batch_num in range(0, train_x.shape[1], 32):
            x = train_x[:, batch_num:batch_num + 32, np.newaxis]
            y = train_y[batch_num:batch_num + 32, np.newaxis]
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)
            output = model.forward(x)
            loss = loss_func(output, y)
            ep_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        now_loss = ep_loss / (train_x.shape[1] / 32)
        print("loss:", now_loss)
        if (abs(now_loss - last_loss) < 0.01):
            last_loss = now_loss
            break
        last_loss = now_loss
    torch.save(model, "./pytorch_model/model_without_cpu_loss_" + str(last_loss) + ".pth")
    # input = Input(shape=(timestep,))
    # re = Reshape(target_shape=(timestep, 1))(input)
    # # lstm = LSTM(120, return_sequences=True)(re)
    # # lstm1 = LSTM(120)(lstm)
    # lstm1 = LSTM(120)(re)
    # flatten = lstm1
    # dense = Dense(10, activation='relu')(flatten)
    # output = Dense(1)(dense)
    # model = Model(inputs=[input], outputs=[output])
    # model.summary()
    # model.compile(loss='mean_absolute_error', optimizer='Adam')
    # checkpoint = ModelCheckpoint(
    #     "./model/predict_temp_ts" + str(timestep) + "_ph" + str(predict_horizon) + "_{val_loss:.2f}.hdf5",
    #     monitor='val_loss',
    #     verbose=1,
    #     save_best_only=True,
    #     mode='min')
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002, patience=3, verbose=1)
    # callbacks_list = [checkpoint, early_stopping]
    # history = model.fit(train_x, train_y, epochs=10000,
    #                     batch_size=128,
    #                     validation_split=0.02, verbose=1, callbacks=callbacks_list)


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
    mix_x = np.concatenate((temp_train_x, cpu_x), axis=1)
    input = Input(shape=(timestep * 2,))
    re = Reshape(target_shape=(timestep * 2, 1))(input)
    lstm = LSTM(120, return_sequences=True)(re)
    lstm1 = LSTM(120)(lstm)
    flatten = lstm1
    dense = Dense(10, activation='relu')(flatten)
    output = Dense(1)(dense)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    checkpoint = ModelCheckpoint(
        "./model/predict_temp_combined_cpu+temp_ts" + str(timestep) + "_ph" + str(
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


def train_model_with_cpu_prediction():
    temp_train_x = []
    temp_train_y = []
    for server in server_list:
        temp = np.array(server.inlet_temp)
        data = strided_app(temp, timestep + predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        y = data[:, -1]
        t_conditioner_outlet_temp = np.array(conditioner_outlet_temp)[48:]
        t_conditioner_outlet_temp = t_conditioner_outlet_temp.reshape(-1, 1)
        x = np.concatenate((x, t_conditioner_outlet_temp), axis=1)
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

    # cpu_pre_model = train_cpu_model(timestep, predict_horizon)
    from keras.models import load_model
    cpu_pre_model = load_model("./model/predict_cpu_ts36_ph12_0.17.hdf5")
    cpu_pre_result = cpu_pre_model.predict(cpu_x)
    mixed_train_x = np.concatenate((temp_train_x, cpu_pre_result), axis=1)
    input = Input(shape=(timestep + 2,))
    re = Reshape(target_shape=(timestep + 2, 1))(input)
    # lstm = LSTM(120, return_sequences=True)(re)
    # lstm1 = LSTM(120)(lstm)
    lstm1 = LSTM(120)(re)
    flatten = lstm1
    dense = Dense(10, activation='relu')(flatten)
    output = Dense(1)(dense)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    checkpoint = ModelCheckpoint(
        "./model/predict_temp_usingcpu+condition_ts" + str(timestep) + "_ph" + str(
            predict_horizon) + "_{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002, patience=3, verbose=1)
    callbacks_list = [checkpoint, early_stopping]
    history = model.fit(mixed_train_x, temp_train_y, epochs=10000,
                        batch_size=128,
                        validation_split=0.02, verbose=1, callbacks=callbacks_list)


def train_cpu_model():
    print("train cpu")
    cpu_x = []
    cpu_y = []
    for server in server_list:
        cpu = np.array(server.cpu)
        data = strided_app(cpu, timestep + predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        y = data[:, -1]
        if isinstance(cpu_x, list):
            cpu_x = x
            cpu_y = y
        else:
            cpu_x = np.concatenate((cpu_x, x), axis=0)
            cpu_y = np.concatenate((cpu_y, y), axis=0)
    input = Input(shape=(timestep,))
    re = Reshape(target_shape=(timestep, 1))(input)
    # lstm = LSTM(120, return_sequences=True)(re)
    # lstm1 = LSTM(120)(lstm)
    lstm1 = LSTM(120)(re)
    flatten = lstm1
    dense = Dense(10, activation='relu')(flatten)
    output = Dense(1)(dense)
    model = Model(inputs=[input], outputs=[output])
    model.summary()
    model.compile(loss='mean_absolute_error', optimizer='Adam')
    checkpoint = ModelCheckpoint(
        "./model/predict_cpu_ts" + str(timestep) + "_ph" + str(predict_horizon) + "_{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.002, patience=3, verbose=1)
    callbacks_list = [checkpoint, early_stopping]
    history = model.fit(cpu_x, cpu_y, epochs=10000,
                        batch_size=128,
                        validation_split=0.02, verbose=1, callbacks=callbacks_list)
    print("train cpu finished")
    return model


def train_model_without_cpu_qat():
    # model_path = ".\pytorch_model\model_without_cpu_loss_0.5797110884647643.pth"
    # model = torch.load(model_path, map_location='cpu')
    # model.named_parameters()
    # model.eval()
    # # print(model)
    # model_int8 = torch.quantization.quantize_dynamic(model,
    #                                                  {torch.nn.LSTM,
    #                                                   torch.nn.Linear},
    #                                                  dtype=torch.qint8)
    # torch.save(model_int8, model_path.split(".pth")[0] + "_quan_qint8.pth")
    # for name, para in model.named_parameters():
    #     print(name, para.size())
    train_x = []
    train_y = []
    for server in server_list:
        inlet_temp = np.array(server.inlet_temp)
        data = strided_app(inlet_temp, timestep + +predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        y = data[:, -1]
        if isinstance(train_x, list):
            train_x = x
            train_y = y
        else:
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)
    print(train_x.shape)
    train_x = train_x.transpose()
    print(train_x.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SingleLSTMQuantization().to(device)
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    # model_fused = torch.quantization.fuse_modules(model, [["l1", "l1Relu"]])
    model_prepared = torch.quantization.prepare_qat(model)

    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    last_loss = 0

    for _ in range(100):
        ep_loss = 0
        for batch_num in range(0, train_x.shape[1], 32):
            x = train_x[:, batch_num:batch_num + 32, np.newaxis]
            y = train_y[batch_num:batch_num + 32, np.newaxis]
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)
            output = model_prepared.forward(x)
            loss = loss_func(output, y)
            ep_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        now_loss = ep_loss / (train_x.shape[1] / 32)
        print("loss:", now_loss)
        if (abs(now_loss - last_loss) < 0.01):
            last_loss = now_loss
            break
        last_loss = now_loss

    model_prepared.eval()
    model_int8 = torch.quantization.convert(model_prepared)
    torch.save(model_int8, "./pytorch_model/model_without_cpu_loss_" + str(last_loss) + "_qat_int8.pth")


if __name__ == "__main__":
    process_data()
    # train_cpu_model()
    # train_model_with_cpu_prediction()
    train_model_without_cpu()
    # train_model_combined_cpu_temp()
    # train_model_without_cpu_qat()
