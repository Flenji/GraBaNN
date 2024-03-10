import torch

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv,SSGConv

from torch_geometric.nn import global_mean_pool


from torch_geometric.datasets import QM9,TUDataset
from torch_geometric.loader import DataLoader,PrefetchLoader
import graph_generation.island_graphs as island_graphs
import graph_generation.RedRatioGraphs as RedRatioGraphs


class Graph_Classification_GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.conv4 = GCNConv(16, 16)
        self.conv5 = GCNConv(16, 16)
        self.lin = Linear(16, 2)

    def forward(self, x, edge_index, batch, edge_weight = None):
        

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x,batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        

        return F.log_softmax(x, dim=1)
    


def model_optimizer_setup(model_constr,device):
    
    model = model_constr().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return model, optimizer


def train(model, optimizer, loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = torch.nn.CrossEntropyLoss()(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader, model):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.loader.dataset)  # Derive ratio of correct predictions.


if __name__=='__main__':
    dataset = RedRatioGraphs.RedRatioGraphs(10000).getDataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_test_split = 0.8
    train_idx = int(len(dataset)*0.8)
    print(train_idx)
    train_dataset = dataset[:train_idx]
    test_dataset = dataset[train_idx:]

    print("dataset downloaded")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    pre_train_loader = PrefetchLoader(train_loader, device)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    pre_test_loader = PrefetchLoader(test_loader, device)
    print("batches created")



    print("done")
    model, optimizer = model_optimizer_setup(Graph_Classification_GCN, device)
    for epoch in range(1, 10):
        train(model, optimizer, pre_train_loader)
        train_acc = test(pre_train_loader, model)
        test_acc = test(pre_test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    torch.save(model, "model/model_red_ratio.pt")
    torch.save(train_loader, "model/test_loader_red_ratio.pt")