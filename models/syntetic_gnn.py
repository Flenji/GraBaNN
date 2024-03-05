import random
import torch

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import synthetic_graph_gen

from torch_geometric.loader import DataLoader,PrefetchLoader



class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(6, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)
        self.conv6 = GCNConv(32, 16)
        self.conv7 = GCNConv(16, 6)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x,edge_index)
        x = F.relu(x)
        x = self.conv3(x,edge_index)
        x = F.relu(x)
        x = self.conv4(x,edge_index)
        x = F.relu(x)
        x = self.conv5(x,edge_index)
        x = F.relu(x)
        x = self.conv6(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv7(x, edge_index)
        

        return F.log_softmax(x, dim=1)
    
class GCN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(6, 16)
        self.conv3 = GCNConv(16, 16)
        self.conv2 = GCNConv(16, 6)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        

        return F.log_softmax(x, dim=1)
    


def model_optimizer_setup(model_constr,device):
    
    model = model_constr().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return model, optimizer


def train(model, optimizer, loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = torch.nn.CrossEntropyLoss()(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader, model):
    model.eval()

    correct = 0
    total_examples = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        #print(len(data))
        out = model(data.x, data.edge_index)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        # Count the number of correct predictions within the batch.
        correct += int((pred == data.y).sum())
        # Track the total number of examples within the batch.
        total_examples += len(data.y)

    return correct / total_examples  # Derive ratio of correct predictions.

 

def collate(dataList):
    return Batch.from_data_list(dataList)

if __name__=='__main__':
    datasetCreator = synthetic_graph_gen.DatasetCreator(10000, 16, 32)
    dataset = datasetCreator.getDataset(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_test_split = 0.8
    train_idx = int(len(dataset)*0.8)
    print(len(dataset))
    train_dataset = dataset[:train_idx]
    test_dataset = dataset[train_idx:]

    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate, batch_size= 512)
    pre_train_loader = PrefetchLoader(train_loader, device)
    test_loader = DataLoader(test_dataset, shuffle=False,collate_fn=collate, batch_size= 512)
    pre_test_loader = PrefetchLoader(test_loader, device)
    print("batches created")
    model, optimizer = model_optimizer_setup(GCN, device)
    for epoch in range(1, 171):
        train(model, optimizer, pre_train_loader)
        train_acc = test(pre_train_loader, model)
        test_acc = test(pre_test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')