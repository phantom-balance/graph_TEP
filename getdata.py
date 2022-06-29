# Normalize using entire dataset or just NOC??
import pandas as pd

def get_data(list, is_train=None):
    noc = pd.read_csv('TEP-profbraatz-dataset/d00.csv')
    mean = noc.mean()
    std = noc.std()

    if is_train==True:
        k = []
        for idx,num in enumerate(list):
            data_path = os.path.join('TEP-profbraatz-dataset/', ('d0' if num < 10 else 'd') + str(num) + ".csv")
            data = pd.read_csv(data_path)
            data_norm = data.copy()
            for i in data_norm:
                data_norm[i] = (data_norm[i]-mean[i])/(std[i])
            if data_path == 'TEP-profbraatz-dataset/d00.csv':
                m = []
                for i in range (500):
                    m.append(0)
                # k.extend([[data],[m]]) # returning the data directly
                k.extend([[data_norm],[m]]) # returning normalized data using mean and standard deviation of training normal operating condition

            else:
                m = []
                for i in range(480):
                    m.append(num)
                # k.extend([[data],[m]]) # returning the data directly
                k.extend([[data_norm],[m]]) # returning normalized data using mean and standard deviation of training normal operating condition

    else:
        k = []
        for idx,num in enumerate(list):
            data_path = os.path.join('TEP-profbraatz-dataset/', ('d0' if num < 10 else 'd') + str(num) + "_te.csv")
            data = pd.read_csv(data_path)
            data_norm = data.copy()
            for i in data_norm:
                data_norm[i] = (data_norm[i]-mean[i])/(std[i])

            if data_path == "TEP-profbraatz-dataset/d00_te.csv":
                m = []
                for i in range(960):
                    m.append(0)
                # k.extend([[data],[m]]) # returning the data directly
                k.extend([[data_norm],[m]]) # returning normalized data using mean and standard deviation of training normal operating condition


            else:
                m = []
                for i in range(160):
                    m.append(0)
                for i in range(160,960):
                    m.append(num)
                # k.extend([[data],[m]]) # returning the data directly
                k.extend([[data_norm],[m]]) # returning normalized data using mean and standard deviation of training normal operating condition

    return k
