from getdata import get_data
from torch_geometric.data import Data
import torch
import pickle

def graph_constructor(num, sequence_length, directed=None, is_train=None):
    print("constructing graph")
    classification_list = []
    new_df_list = []
    graph_data_list = []
    Lis = get_data(num, is_train=is_train)

    k = 0
    for i in range(len(Lis)):
        if i%2 != 0:
            list_class = Lis[i]
            list_instance = list_class[0]
            k = k + len(list_instance) - sequence_length + 1
        else:
            pass
    length = k

    for i in range(len(Lis)):
        if i % 2 != 0:
            list_class = Lis[i]
            list_instance = list_class[0]
            temp_list_instance = list_instance.copy()
            for j in range(sequence_length-1):
                temp_list_instance.pop(0)
            classification_list = classification_list + temp_list_instance

        else:
            df_class = Lis[i]
            df_instance = df_class[0]
            df_instance = df_instance.values.tolist()
            temp_df_list = df_instance.copy()
            for i in range(len(df_instance)-sequence_length+1):
                for j in range(sequence_length):
                    new_df_list.append(temp_df_list[j])
                temp_df_list.pop(0)
    
    classification_tensor = torch.tensor(classification_list)
    # print(classification_tensor.shape)
    df_tensor = torch.tensor(new_df_list)
    df_tensor = df_tensor.reshape(length, sequence_length,52)
    # print(df_tensor.shape)
    for i in range(length):
        node_initialization = torch.zeros(sequence_length,30)
        df_tensor_temp = df_tensor[i]
        df_new = torch.cat((df_tensor_temp, node_initialization), dim=1)
        if i == 0:
            df_tensor_new = df_new
        else:
            df_tensor_new = torch.cat((df_tensor_new, df_new), dim=0)
    
    df_tensor_new = df_tensor_new.reshape(length,sequence_length,82)
    # print(df_tensor_new.shape)


    if directed==True:
        edge_index = pickle.load(open("processed_data/directed_adjacency_list.p", "rb"))
    else :
        edge_index = pickle.load(open("processed_data/undirected_adjacency_list.p", "rb"))

    for k in range(length):
        data = df_tensor_new[k].t()

        data_instance = Data(x = data,
                            edge_index = edge_index,
                            y = classification_tensor[k])
        graph_data_list.append(data_instance)
    return graph_data_list, length


# Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# dat, lens = graph_constructor(num=Type, sequence_length=4, directed=False, is_train=True)
