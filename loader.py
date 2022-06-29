from torch.utils.data import Dataset

class TEP(Dataset):
    def __init__(self, graph_data, length):
        self.graph_data = graph_data
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.graph_data[index]

