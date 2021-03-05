import numpy as np

import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["DGLBACKEND"] = "pytorch"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = "./singlerack_all.csv"
server_list = []
conditioner_outlet_temp = []
conditioner_inlet_temp = []

timestep = 36
predict_horizon = 12
machines_num = 15


class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.g = g
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(self.g, h)
        h = F.elu(h)
        h = h.view(h.size(0), -1)
        h = self.layer2(self.g, h)
        return h


class GATCombineLSTM(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GATCombineLSTM, self).__init__()
        self.g = g
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 5)

    def forward(self, h):
        h = self.layer1(self.g, h)
        h = F.elu(h)
        h = h.view(h.size(0), -1)
        h = self.layer2(self.g, h)
        return h


class LSTMCombinedGAT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMCombinedGAT, self).__init__()
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.lstm = nn.LSTM(input_size=input_size
                            , hidden_size=hidden_size
                            , num_layers=num_layers)
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
        x, (hn, cn) = self.lstm(x)
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


def build_homo_graph():
    neighbour_num = 1
    cooling_num = 1
    u_list = []
    v_list = []
    for i in range(machines_num):
        for nei in range(1, neighbour_num + 1):
            last = i - nei
            next = i + nei
            if last >= 0:
                u_list.append(i)
                v_list.append(last)
            if next < 15:
                u_list.append(i)
                v_list.append(next)
    for i in range(machines_num):
        for j in range(machines_num, machines_num + cooling_num):
            u_list.append(i)
            v_list.append(j)
            u_list.append(j)
            v_list.append(i)
    return dgl.graph((u_list, v_list), idtype=torch.int32)


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


def train_gnn_model_combined_cpu():
    process_data()
    graph = build_homo_graph()
    #一个邻居准确率1.24
    graph = dgl.add_self_loop(graph).to(device)
    train_x = []
    train_y = []
    cpu_x = []
    for server in server_list:
        inlet_temp = np.array(server.inlet_temp)
        data = strided_app(inlet_temp, timestep + predict_horizon + 1, 1)
        cpu = np.array(server.cpu)
        data_cpu = strided_app(cpu, timestep + predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        x_cpu = data_cpu[:, :-1 - predict_horizon]
        x = x[:, :, np.newaxis]
        x_cpu = x_cpu[:, :, np.newaxis]
        x = np.concatenate((x, x_cpu), axis=2)
        y = data[:, -1]
        if isinstance(train_x, list):
            train_x = x[np.newaxis, :, :, :]
            train_y = y[np.newaxis, :]
        else:
            x = x[np.newaxis, :, :, :]
            y = y[np.newaxis, :]
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)

    conditioner_outlet_temp_t = strided_app(np.array(conditioner_outlet_temp), timestep + predict_horizon + 1, 1)
    conditioner_outlet_temp_t_x = conditioner_outlet_temp_t[np.newaxis, :, :-1 - predict_horizon]
    conditioner_outlet_temp_t_x_for_combined = conditioner_outlet_temp_t_x[:, :, :, np.newaxis]
    # conditioner_outlet_temp_t_y = conditioner_outlet_temp_t[np.newaxis, :, -1]
    conditioner_outlet_temp_t_x_for_combined = conditioner_outlet_temp_t_x_for_combined.repeat(15, axis=0)

    # train_x_lstm:[mahines_nums,data_len,timestep,(temp,cpu,conditioner_out_temp)]
    train_x_lstm = np.concatenate((train_x, conditioner_outlet_temp_t_x_for_combined), axis=3)

    # train_x_gnn:[mahines_nums,data_len,timestep]
    train_x_gnn = train_x[:, :, :, 0]
    # train_x_gnn:[mahines_nums+1,data_len,timestep]
    train_x_gnn = np.concatenate((train_x_gnn, conditioner_outlet_temp_t_x), axis=0)
    # train_x_gnn:[data_len,mahines_nums+1,timestep]
    train_x_gnn = train_x_gnn.transpose((1, 0, 2))

    net_gat = GATCombineLSTM(graph, in_dim=train_x_gnn.shape[2], hidden_dim=50, out_dim=2, num_heads=3).to(device)
    net_lstm = LSTMCombinedGAT(input_size=13, hidden_size=120, num_layers=1).to(device)
    optimizer_gnn = torch.optim.Adam(net_gat.parameters(), lr=1e-3)
    optimizer_lstm = torch.optim.Adam(net_lstm.parameters(), lr=1e-3)
    train_x_gnn = torch.FloatTensor(train_x_gnn).to(device)
    train_x_lstm = torch.FloatTensor(train_x_lstm).to(device)
    train_y = torch.FloatTensor(train_y).to(device)
    best_loss = 10e3
    for epoch in range(30):
        loss_sum = 0
        for i in range(train_x_gnn.shape[0]):
            for machine_id in range(machines_num):
                feature = train_x_gnn[i, :, :]
                y_gnn = net_gat(feature)
                y_gnn = y_gnn.view(y_gnn.size(0), -1)
                gnn_output = y_gnn[machine_id, :]
                gnn_output = gnn_output.repeat(36, 1)
                input_lstm = train_x_lstm[machine_id, i, :, :]
                input = torch.cat((gnn_output, input_lstm), dim=1)
                input = torch.unsqueeze(input, 1)
                y_lstm = net_lstm(input).view(-1)
                y_ground = train_y[machine_id, i].view(-1)
                # print(y.size(),y_ground.size())
                loss = F.mse_loss(y_lstm, y_ground)
                # print(loss)
                # optimizer_gnn.zero_grad()
                optimizer_lstm.zero_grad()
                loss.backward()
                optimizer_gnn.step()
                optimizer_lstm.step()
                loss_sum += loss.data
            if i % 100 == 0:
                print(loss_sum / i / machines_num)
        avg_loss = loss_sum / train_x_gnn.shape[0] / machines_num
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net_gat, "./dgl_with_cool_w_cpu_gnn_clstm.pth")
            torch.save(net_lstm, "./dgl_with_cool_w_cpu_lstm_cgnn.pth")
            print("avg_loss", avg_loss)


def train_gnn_model_wo_cpu():
    process_data()
    graph = build_homo_graph()
    graph = dgl.add_self_loop(graph).to(device)
    # graph.ndata["w"] = torch.ones(graph.num_nodes(), timestep + predict_horizon)
    # graph.edata["w"] = torch.ones(graph.num_edges(), 3)
    train_x = []
    train_y = []
    for server in server_list:
        inlet_temp = np.array(server.inlet_temp)
        data = strided_app(inlet_temp, timestep + predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        y = data[:, -1]
        if isinstance(train_x, list):
            train_x = x[np.newaxis, :, :]
            train_y = y[np.newaxis, :]
        else:
            x = x[np.newaxis, :, :]
            y = y[np.newaxis, :]
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)

    conditioner_outlet_temp_t = strided_app(np.array(conditioner_outlet_temp), timestep + predict_horizon + 1, 1)
    conditioner_outlet_temp_t_x = conditioner_outlet_temp_t[np.newaxis, :, :-1 - predict_horizon]
    conditioner_outlet_temp_t_y = conditioner_outlet_temp_t[np.newaxis, :, -1]
    train_x = np.concatenate((train_x, conditioner_outlet_temp_t_x), axis=0)
    train_y = np.concatenate((train_y, conditioner_outlet_temp_t_y), axis=0)

    train_x = train_x.transpose((1, 0, 2))
    train_y = train_y.transpose((1, 0))

    net = GAT(graph, in_dim=train_x.shape[2], hidden_dim=50, out_dim=1, num_heads=3).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    best_loss = 10e3
    for epoch in range(30):
        loss_sum = 0
        for i in range(train_x.shape[0]):
            feature = torch.FloatTensor(train_x[i, :, :]).to(device)
            y = net(feature).view(-1)
            y_ground = torch.FloatTensor(train_y[i, :]).view(-1).to(device)
            # print(y.size(),y_ground.size())
            loss = F.mse_loss(y, y_ground)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.data
        avg_loss = loss_sum / train_x.shape[0]
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net, "./dgl_with_cool_wo_cpu.pth")
            print(avg_loss)
    print(graph)


def train_model_without_cpu():
    process_data()
    train_x = []
    train_y = []
    for server in server_list:
        inlet_temp = np.array(server.inlet_temp)
        data = strided_app(inlet_temp, timestep + predict_horizon + 1, 1)
        x = data[:, :-1 - predict_horizon]
        y = data[:, -1]
        if isinstance(train_x, list):
            train_x = x
            train_y = y
        else:
            train_x = np.concatenate((train_x, x), axis=0)
            train_y = np.concatenate((train_y, y), axis=0)
    # print(train_x.shape)
    train_x = train_x.transpose()
    print(train_x.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    last_loss = 1e8
    tor = 0
    while True:
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
        if now_loss < last_loss:
            last_loss = now_loss
            import os
            # os.makedirs("./pytorch_model")
            torch.save(model, "./model_without_cpu_loss_" + str(last_loss) + ".pth")
            if (abs(now_loss - last_loss) < 0.01):
                break


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
    # process_data()
    # train_cpu_model()
    # train_model_with_cpu_prediction()
    # train_model_without_cpu()
    # train_model_combined_cpu_temp()
    # train_model_without_cpu_qat()
    # train_gnn_model_wo_cpu()
    train_gnn_model_combined_cpu()
