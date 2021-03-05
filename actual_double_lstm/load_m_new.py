from keras.models import load_model
import numpy as np
import random
import time


def process_lstm(layer, weight_list):
    prefix = layer.name.upper() + "_"
    lstm_kernel_weights, re_kernel_weights, bias_weights = layer.get_weights()
    single_kernel_length = re_kernel_weights.shape[0]
    lstm_kernel_weights_i = lstm_kernel_weights[:, :single_kernel_length]
    weight_list.append((prefix + "kernel_i".upper(), lstm_kernel_weights_i))
    lstm_kernel_weights_f = lstm_kernel_weights[:, single_kernel_length:single_kernel_length * 2]
    weight_list.append((prefix + "kernel_f".upper(), lstm_kernel_weights_f))
    lstm_kernel_weights_c = lstm_kernel_weights[:, single_kernel_length * 2:single_kernel_length * 3]
    weight_list.append((prefix + "kernel_c".upper(), lstm_kernel_weights_c))
    lstm_kernel_weights_o = lstm_kernel_weights[:, single_kernel_length * 3:single_kernel_length * 4]
    weight_list.append((prefix + "kernel_o".upper(), lstm_kernel_weights_o))
    re_kernel_weights_i = re_kernel_weights[:, :single_kernel_length]
    weight_list.append((prefix + "re_kernel_i".upper(), re_kernel_weights_i))
    re_kernel_weights_f = re_kernel_weights[:, single_kernel_length:single_kernel_length * 2]
    weight_list.append((prefix + "re_kernel_f".upper(), re_kernel_weights_f))
    re_kernel_weights_c = re_kernel_weights[:, single_kernel_length * 2:single_kernel_length * 3]
    weight_list.append((prefix + "re_kernel_c".upper(), re_kernel_weights_c))
    re_kernel_weights_o = re_kernel_weights[:, single_kernel_length * 3:single_kernel_length * 4]
    weight_list.append((prefix + "re_kernel_o".upper(), re_kernel_weights_o))
    bias_weights_i = bias_weights[:single_kernel_length]
    weight_list.append((prefix + "bias_i".upper(), bias_weights_i))
    bias_weights_f = bias_weights[single_kernel_length:single_kernel_length * 2]
    weight_list.append((prefix + "bias_f".upper(), bias_weights_f))
    bias_weights_c = bias_weights[single_kernel_length * 2:single_kernel_length * 3]
    weight_list.append((prefix + "bias_c".upper(), bias_weights_c))
    bias_weights_o = bias_weights[single_kernel_length * 3:single_kernel_length * 4]
    weight_list.append((prefix + "bias_o".upper(), bias_weights_o))


def process_dense(layer, weight_list):
    prefix = layer.name.upper() + "_"
    kernel_weights, bias_weights = layer.get_weights()
    weight_list.append((prefix + "kernel".upper(), kernel_weights))
    weight_list.append((prefix + "bias".upper(), bias_weights))


model = load_model("./model/predict_cpu_ts36_ph12_0.17.hdf5")
print(model.summary())
weight_list = []
for layer in model.layers:
    if "lstm" not in layer.name and "dense" not in layer.name:
        continue
    if "lstm" in layer.name:
        process_lstm(layer, weight_list)
    if "dense" in layer.name:
        process_dense(layer, weight_list)
for name, weight in weight_list:
    print(name)
