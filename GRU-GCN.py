# Requires batch training implementation, requires multiple feature from single node if possible

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU

import os
from loader import TEP
from construct_graph import graph_constructor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_nodes = 82
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22

sequence_length = 15
node_features=15
learning_rate = 0.0003
num_epochs = 1
batch_size = 5
embedding_size = 15
directed = False

load_model = False
save_model = True

if directed==True:
    if os.path.isfile(f"processed_data/Data_directed/{sequence_length}.pt"):
        print("graph data exists")
        graph_dat = torch.load(f"processed_data/Data_directed/{sequence_length}.pt")
    else:
        print("creating graph data")
        train_set, length_train = graph_constructor(num=Type, sequence_length=sequence_length, directed=True, is_train=True)
        train_graph = TEP(train_set, length_train)
        test_set, length_test = graph_constructor(num=Type, sequence_length=sequence_length, directed=True, is_train=False)
        test_graph = TEP(test_set, length_test)
        graph_dat = {'train': train_graph,
                     'test': test_graph}
        torch.save(graph_dat, f"processed_data/Data_directed/{sequence_length}.pt")

else:
    if os.path.isfile(f"processed_data/Data_undirected/{sequence_length}.pt"):
        print("graph data exists")
        graph_dat = torch.load(f"processed_data/Data_undirected/{sequence_length}.pt")
    else:
        print("creating graph data")
        train_set, length_train = graph_constructor(num=Type, sequence_length=sequence_length, directed=False, is_train=True)
        train_graph = TEP(train_set, length_train)
        test_set, length_test = graph_constructor(num=Type, sequence_length=sequence_length, directed=False, is_train=False)
        test_graph = TEP(test_set, length_test)
        graph_dat = {'train': train_graph,
                     'test': test_graph}
        torch.save(graph_dat, f"processed_data/Data_undirected/{sequence_length}.pt")


class GRUGCN(nn.Module):
    def __init__(self, node_features):
        super().__init__()
        self.recurrent = GConvGRU(node_features,30,1)
        self.linear = nn.Linear(30*num_nodes,22)

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = h.view(-1, 30*num_nodes)
        h = self.linear(h)
        return h

model = GRUGCN(node_features=sequence_length)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# train_loader = DataLoader(dataset=graph_dat['train'], batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(dataset=graph_dat['test'], batch_size=batch_size, shuffle=False)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device=device)
            scores = model(batch.x, batch.edge_index)

            _, prediction = scores.max(1)
            num_correct += (prediction==batch.y).sum()
            num_samples += prediction.size(0)
        print(f'Got {num_correct}/{num_samples} correct, prediction rate={float(num_correct)/float(num_samples)*100:.3f}')
    model.train()


for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(graph_dat["train"]):
        # batch = next(iter(train_loader))
        batch = batch.to(device)
        scores = model(batch.x, batch.edge_index)
        batch.y = batch.y.item()
        batch.y = torch.tensor([batch.y])
        loss = criterion(scores, torch.tensor(batch.y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch:{epoch}| |Batch_idx:{batch_idx}")
        if save_model==True:
            if (batch_idx+1) % 1500 == 0:
                # checkpoint = {'state_dict': model.state_dict(),
                #             'optimizer': optimizer.state_dict()}
                # save_checkpoint(checkpoint)
                print("checking accuracy on Testing Set")
                check_accuracy(graph_dat["test"], model)
                print("checking accuracy on Training Set")
                check_accuracy(graph_dat["train"], model)


print("checking accuracy on Testing Set")
check_accuracy(graph_dat["test"], model)
print("checking accuracy on Training Set")
check_accuracy(graph_dat["train"], model)

