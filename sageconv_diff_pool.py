from construct_graph import graph_constructor
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
# from torch_geometric.loader import DataLoader
from torch_geometric.loader import DenseDataLoader
from loader import TEP 
from math import ceil
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj = pickle.load(open("processed_data/undirected_adjacency_matrix.p", "rb"))
adj = torch.from_numpy(adj).float()

max_nodes = num_nodes = 82
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22

num_features = sequence_length = 15
learning_rate = 0.003
num_epochs = 1
batch_size = 32
embedding_size = 15
directed = False
node_red_ratio = 0.6

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

train_loader = DenseDataLoader(dataset=graph_dat['train'], batch_size=batch_size, shuffle=True)
test_loader = DenseDataLoader(dataset=graph_dat['test'], batch_size=batch_size, shuffle=True)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, lin=True):
        super().__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)
        
        if lin is True:
            self.lin = torch.nn.Linear(2*hidden_channels + out_channels, out_channels)
        else:
            self.lin = None

    def forward(self, x, adj):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.conv1(x0, adj).relu()
        x2 = self.conv2(x1, adj).relu()
        x3 = self.conv3(x2, adj).relu()

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()
        
        return x

# mod = GNN(15, 10, 10, lin=False)
# mod2 = GNN(15, 10, 10)
# x = torch.rand(32,82,15).float()
# print(mod(x, adj.float()).shape)
# print(mod2(x, adj.float()).shape)

class DiffPool(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(node_red_ratio*max_nodes)
        self.gnn1_pool = GNN(sequence_length, num_nodes*2, num_nodes)
        self.gnn1_embed = GNN(sequence_length, 15, 15, lin=False)

        num_nodes = ceil(node_red_ratio*num_nodes)
        self.gnn2_pool = GNN(3*15, 15, num_nodes)
        self.gnn2_embed = GNN(3*15, 15, 15, lin=False)
        print("hot")
        self.gnn3_embed = GNN(3*15, 15, 15, lin=False)

        self.lin1 = torch.nn.Linear(3*15, 30)
        self.lin2 = torch.nn.Linear(30, num_classes)

    def forward(self, x, adj):
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)

        x, adj, _, _ = dense_diff_pool(x, adj, s)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, _, _ = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x

model = DiffPool().to(device)
print(model)
print("Number of parametersl", sum(p.numel() for p in model.parameters()))
# x = torch.rand(32, 82, 15)
# out = model(x, adj)
# print(out.shape)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if directed==True:
    def save_checkpoint(state, filename=f"processed_data/sageconv_diffpool-{sequence_length}_directed.pth.tar"):
        print("__Saving Checkpoint__")
        torch.save(state, filename)

else:
    def save_checkpoint(state, filename=f"processed_data/sageconv_diffpool-{sequence_length}_undirected.pth.tar"):
        print("__Saving Checkpoint__")
        torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("__Loading Checkpoint__")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = model(batch.x, adj)

            _, prediction = scores.max(1)
            num_correct += (prediction==batch.y).sum()
            num_samples += prediction.size(0)
        print(f'Got {num_correct}/{num_samples} correct, prediction rate= {float(num_correct)/float(num_samples)*100:.3f}')
    model.train()

if load_model == True and directed == True:
    load_checkpoint(torch.load(f"processed_data/sageconv_diffpool-{sequence_length}_directed.pth.tar", map_location=device))

elif load_model == True and directed == False:
    load_checkpoint(torch.load(f"processed_data/sageconv_diffpool-{sequence_length}_undirected.pth.tar", map_location=device))

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        scores = model(batch.x, adj)
        loss = criterion(scores, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if save_model == True:
            if batch_idx % 25 == 0:
                print(f"Epoch:{epoch}| |Batch_idx:{batch_idx}")
                checkpoint = {'state_dict': model.state_dict(),
                              'optimizer': optimizer.state_dict()}

                save_checkpoint(checkpoint)
                print("checking accuracy on Testing Set")
                check_accuracy(test_loader, model)
                print("checking accuracy on Training Set")
                check_accuracy(train_loader, model)
