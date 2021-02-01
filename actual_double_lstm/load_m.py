from keras.models import load_model
import numpy as np
import random
import time

cpu_model = load_model("./model/predict_cpu_ts36_ph12_0.17.hdf5")


def cal_model_time():
    # input_list=[]
    # for _ in range(36):
    #     input_list.append(random.randint(1,100))
    # input=np.array(input_list)
    input=np.array([0.3 for _ in range(36)])
    input = input.reshape(-1, 36)
    times=1
    cal_time_sum=0
    for _ in range(times):
        input = np.array([random.randint(1,100) for _ in range(36)])
        input = input.reshape(-1, 36)
        start_time = time.time_ns()
        res = cpu_model.predict(input)
        print(res)
        end_time = time.time_ns()
        cal_time_sum+=((end_time - start_time)/10e6)
    print("time_ms:", cal_time_sum/times)



def print_model_weight(file_path):
    lstm_layer = cpu_model.get_layer('lstm')
    dense_layer1 = cpu_model.get_layer('dense')
    dense_layer2 = cpu_model.get_layer('dense_1')
    weights_list = []
    lstm_kernel_weights, re_kernel_weights, bias_weights = lstm_layer.get_weights()
    # dense1_kernel_weights,dense1_bias_weights=dense_layer1.get_weights()
    # dense2_kernel_weights, dense2_bias_weights = dense_layer2.get_weights()
    single_kernel_length = re_kernel_weights.shape[0]
    lstm_kernel_weights_i = lstm_kernel_weights[:, :single_kernel_length]
    lstm_kernel_weights_f = lstm_kernel_weights[:, single_kernel_length:single_kernel_length * 2]
    lstm_kernel_weights_c = lstm_kernel_weights[:, single_kernel_length * 2:single_kernel_length * 3]
    lstm_kernel_weights_o = lstm_kernel_weights[:, single_kernel_length * 3:single_kernel_length * 4]
    re_kernel_weights_i = re_kernel_weights[:, :single_kernel_length]
    re_kernel_weights_f = re_kernel_weights[:, single_kernel_length:single_kernel_length * 2]
    re_kernel_weights_c = re_kernel_weights[:, single_kernel_length * 2:single_kernel_length * 3]
    re_kernel_weights_o = re_kernel_weights[:, single_kernel_length * 3:single_kernel_length * 4]
    bias_weights_i = bias_weights[:single_kernel_length]
    bias_weights_f = bias_weights[single_kernel_length:single_kernel_length * 2]
    bias_weights_c = bias_weights[single_kernel_length * 2:single_kernel_length * 3]
    bias_weights_o = bias_weights[single_kernel_length * 3:single_kernel_length * 4]
    weights_list.extend([lstm_kernel_weights_i, lstm_kernel_weights_f, lstm_kernel_weights_c,
                         lstm_kernel_weights_o, re_kernel_weights_i, re_kernel_weights_f,
                         re_kernel_weights_c, re_kernel_weights_o, bias_weights_i,
                         bias_weights_f, bias_weights_c, bias_weights_o])
    weights_list.extend(dense_layer1.get_weights())
    weights_list.extend(dense_layer2.get_weights())
    np.set_printoptions(threshold=np.inf)
    weights_str_list=[]
    for nda in weights_list:
        weights_str = np.array2string(nda, suppress_small=True, separator=",",floatmode="maxprec_equal")
        weights_str = weights_str.replace("\n", "")
        weights_str=weights_str.replace("[","")
        weights_str=weights_str.replace("]","")
        weights_str = weights_str.replace(",", ",\n")
        weights_str = "{" + weights_str + "}"
        weights_str_list.append(weights_str)
    with open(file_path, "w") as f:
        f.write("""#ifndef _WAB_
#define _WAB_\n
        """)
        f.write("#define UNITS "+str(single_kernel_length)+"\n")
        f.write("#define DENSE1_UNITS " + str(dense_layer1.get_weights()[1].shape[0]) + "\n")
        f.write("#define DENSE2_UNITS " + str(dense_layer2.get_weights()[1].shape[0]) + "\n")
        f.write("#define INPUT_FEATURE "+str(lstm_kernel_weights.shape[0])+"\n")
        f.write("""#define KERNEL_TYPE double
#define RE_KERNEL_TYPE double
#define BIAS_TYPE double\n
        """)
        f.write("KERNEL_TYPE m_kernel_i[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[0])
        f.write(";\n")
        f.write("KERNEL_TYPE m_kernel_f[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[1])
        f.write(";\n")
        f.write("KERNEL_TYPE m_kernel_c[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[2])
        f.write(";\n")
        f.write("KERNEL_TYPE m_kernel_o[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[3])
        f.write(";\n")
        f.write("RE_KERNEL_TYPE m_re_kernel_i[UNITS][UNITS]=")
        f.write(weights_str_list[4])
        f.write(";\n")
        f.write("RE_KERNEL_TYPE m_re_kernel_f[UNITS][UNITS]=")
        f.write(weights_str_list[5])
        f.write(";\n")
        f.write("RE_KERNEL_TYPE m_re_kernel_c[UNITS][UNITS]=")
        f.write(weights_str_list[6])
        f.write(";\n")
        f.write("RE_KERNEL_TYPE m_re_kernel_o[UNITS][UNITS]=")
        f.write(weights_str_list[7])
        f.write(";\n")
        f.write("BIAS_TYPE m_bias_i[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[8])
        f.write(";\n")
        f.write("BIAS_TYPE m_bias_f[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[9])
        f.write(";\n")
        f.write("BIAS_TYPE m_bias_c[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[10])
        f.write(";\n")
        f.write("BIAS_TYPE m_bias_o[INPUT_FEATURE][UNITS]=")
        f.write(weights_str_list[11])
        f.write(";\n")
        f.write("KERNEL_TYPE dense1_kernel[UNITS][DENSE1_UNITS]=")
        f.write(weights_str_list[12])
        f.write(";\n")
        f.write("BIAS_TYPE dense1_bias[INPUT_FEATURE][DENSE1_UNITS]=")
        f.write(weights_str_list[13])
        f.write(";\n")
        f.write("KERNEL_TYPE dense2_kernel[DENSE1_UNITS][DENSE2_UNITS]=")
        f.write(weights_str_list[14])
        f.write(";\n")
        f.write("BIAS_TYPE dense2_bias[INPUT_FEATURE][DENSE2_UNITS]=")
        f.write(weights_str_list[15])
        f.write(";\n")
        f.write("#endif")

if __name__ == '__main__':
    cal_model_time()
    # print_model_weight("./weight_and_bias.h")
