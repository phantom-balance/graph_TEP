from construct_graph import graph_constructor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.loader import DataLoader
from loader import TEP
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_nodes = 82
sequence_length = 20
Type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
num_classes = 22
learning_rate = 0.003
num_epochs = 2000
batch_size = 29
load_model = False
save_model = True
embedding_size = 20
directed = False

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


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(sequence_length, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size*num_nodes,300)
        self.dropout = nn.Dropout(p=0.5)
        self.out = nn.Linear(300,num_classes)

    def forward(self, x, edge_index, batch_index):
        # print("0",x.shape)
        updated_node = self.conv1(x, edge_index)
        updated_node = F.relu(updated_node)
        # print("1",updated_node.shape)
        updated_node = self.conv2(updated_node, edge_index)
        updated_node = F.relu(updated_node)
        # print("2",updated_node.shape)
        # latent_feature = gmp(updated_node, batch_index)
        # latent_feature = updated_node.view(batch_size, num_nodes*embedding_size)
        latent_feature = updated_node.view(-1, num_nodes*embedding_size)
        # print("3",latent_feature.shape)
        latent_feature = self.fc1(latent_feature)
        latent_feature = F.relu(latent_feature)
        # print("4",latent_feature.shape)
        latent_feature = self.dropout(latent_feature)
        out = self.out(latent_feature)
        # print("5",out.shape)

        return out
    
model = GCN().to(device)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

def save_checkpoint(state, filename=f"processed_data/GCN_TEP-{sequence_length}.pth.tar"):
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
            batch = batch.to(device=device)
            scores = model(batch.x, batch.edge_index, batch.batch)

            _, prediction = scores.max(1)
            num_correct += (prediction==batch.y).sum()
            num_samples += prediction.size(0)
        print(f'Got {num_correct}/{num_samples} correct, prediction rate={float(num_correct)/float(num_samples)*100:.3f}')
    model.train()

train_loader = DataLoader(dataset=graph_dat['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=graph_dat['test'], batch_size=batch_size, shuffle=True)

if load_model == True:
    load_checkpoint(torch.load(f"processed_data/GCN_TEP-{sequence_length}.pth.tar"))

for epoch in range(num_epochs):
    batch = next(iter(train_loader))
    batch = batch.to(device)
    scores = model(batch.x, batch.edge_index, batch.batch)
    loss = criterion(scores, batch.y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if save_model==True:
        if epoch % 100 == 0:
            checkpoint = {'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
            print("checking accuracy on Testing Set")
            check_accuracy(test_loader, model)
            print("checking accuracy on Training Set")
            check_accuracy(train_loader, model)

def summary_return(DATA, is_directed=None):
    if is_directed==True:
        graph_dat = torch.load(f"processed_data/Data_directed/{sequence_length}.pt")
    else:
        graph_dat = torch.load(f"processed_data/Data_undirected/{sequence_length}.pt")
    
    graph_train_set = graph_dat['train']
    graph_test_set = graph_dat['test']

    Train_loader = DataLoader(dataset=graph_train_set, batch_size=50, shuffle=False)
    Test_loader = DataLoader(dataset=graph_test_set, batch_size=50, shuffle=False)

    load_checkpoint(torch.load(f"processed_data/GCN_TEP-{sequence_length}.pth.tar"))

    y_true = []
    y_pred = []
    y_prob = torch.double

    if DATA == "train":
        with torch.no_grad():
            for batch_idx, batch in enumerate(Train_loader):
                batch = batch.to(device=device)
                scores = model(batch.x, batch.edge_index, batch.batch)
                prob = nn.Softmax(dim=1)
                y_prob_temp = prob(scores)
                if batch_idx == 0:
                    y_prob = y_prob_temp
                else:
                    y_prob = torch.cat((y_prob, y_prob_temp), dim=0)
                _, predictions = scores.max(1)
                y_pred.extend(predictions)
                y_true.extend(batch.y)
    elif DATA == "test":
        with torch.no_grad():
            for batch_idx, batch in enumerate(Test_loader):
                batch = batch.to(device=device)
                scores = model(batch.x, batch.edge_index, batch.batch)
                prob = nn.Softmax(dim=1)
                y_prob_temp = prob(scores)
                if batch_idx == 0:
                    y_prob = y_prob_temp
                else:
                    y_prob = torch.cat((y_prob, y_prob_temp), dim=0)
                _, predictions = scores.max(1)
                y_pred.extend(predictions)
                y_true.extend(batch.y)
    else:
        print("enter either test or false")

    return y_true, y_pred, y_prob
    