import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE
from loader import TEP
from construct_graph import graph_constructor
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_nodes = 82
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22

sequence_length = 15
learning_rate = 0.003
num_epochs = 4
batch_size = 32
embedding_size = 15
directed = False

load_model = True
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

class GCNAE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(sequence_length, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, 4)

    def forward(self, x, edge_index, batch_index):
        updated_node = self.conv1(x, edge_index)
        updated_node = F.relu(updated_node)
        updated_node = self.conv2(updated_node, edge_index)
        updated_node = F.relu(updated_node)
        updated_node = self.conv3(updated_node, edge_index)

        return updated_node

model = GAE(GCNAE(sequence_length)).to(device)

print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(state, filename=f"processed_data/GAE_TEP-{sequence_length}.pth.tar"):
    print("__Saving Checkpoint__")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("__Loading Checkpoint__")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

train_loader = DataLoader(dataset=graph_dat['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=graph_dat['test'], batch_size=batch_size, shuffle=True)

if load_model == True:
    load_checkpoint(torch.load(f"processed_data/GAE_TEP-{sequence_length}.pth.tar", map_location=device))

for epoch in range(num_epochs):
  for batch_idx, batch in enumerate(train_loader):
    batch=batch.to(device)
    model.train()
    z = model.encode(batch.x, batch.edge_index, batch.batch)
    loss = model.recon_loss(z, batch.edge_index)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
      print(f"Epoch:{epoch}||Batch_idx:{batch_idx}||Loss:{loss}")
      checkpoint = {'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}
      save_checkpoint(checkpoint)