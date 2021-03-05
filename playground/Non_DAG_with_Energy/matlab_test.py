import os
import time
from SixSigmaDCSDK.engine import Engine
engine=Engine()
front1List = []
back5List = []
fanspeed = 100
base_path= "/playground/Non_DAG_SixSigmaDC/5Server\Baseline\SolverExchange\\"
if os.path.exists(base_path+"out.csv"):
    os.remove(base_path+"out.csv")
if os.path.exists(base_path + "in.csv"):
    os.remove(base_path + "in.csv")
while True:
    if os.path.exists(base_path+"out.csv"):
        # print("finde out csv")
        if (engine.readCFDOutFile(base_path+"out.csv")):
            # print("getin out csv")
            front1List.append(engine.getSolutionAttr("Sensor", "front1", "Value").value)
            back5List.append(engine.getSolutionAttr("Sensor", "back5", "Value").value)
            fanspeed = fanspeed - 1 if fanspeed > 1 else 0
            print("fanspeed:",fanspeed)
            engine.setSolutionAttr("ACU", "ACU01", "FanSpeed", str(fanspeed),"%")
            engine.writeCFDInFile(base_path+'in.csv')
            time.sleep(1)
        else:
            time.sleep(1)
    else:
        time.sleep(1)
