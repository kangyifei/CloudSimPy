from keras.models import load_model
import numpy as np
from keras.layers import LSTM
inputs=[]
for _ in range(36):
    inputs.append(0.3)
model=load_model("./model/predict_cpu_ts36_ph12_0.17.hdf5")
output=model.predict(np.array(inputs).reshape(-1,36))
print(output)