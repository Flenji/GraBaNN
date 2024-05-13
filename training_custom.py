from explainers.gnninterpreter import *

import torch
from tqdm.auto import trange

from torch_geometric.loader import DataLoader
from graph_generation.MultiGraphs import MultiGraphs
import graph_generation.RedRatioGraphs as RedRatioGraphs
import graph_generation.DuoSet as DouGraphs
from torch import nn

from torchmetrics import F1Score

import numpy as np

from libraries.dataLoaderWrapper import GNNInterpreterLoaderWrapper

def model_optimizer_setup(model_constr,device):
    
    model = model_constr(node_features=3,  num_classes =2, hidden_channels = 32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return model, optimizer


def train(model, optimizer, loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
         out = model(batch= data)  # Perform a single forward pass.
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


def fit_model(model, loader, batch_size=32, lr=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for batch in loader:
        model.zero_grad()  # Clear gradients.
        out = model(batch)  # Perform a single forward pass.
        loss = criterion(out['logits'], batch.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        losses.append(loss.item())
    return np.mean(losses)

def evaluate_model(dataloader,model, num_classes):
    f1 = F1Score(task="multiclass", num_classes=num_classes, average=None)
    model.eval()
    for batch in dataloader:
        f1(model(batch)['logits'], batch.y)
    return dict(zip(range(num_classes), f1.compute().tolist()))

if __name__=='__main__':
    dataset = DouGraphs.DuoSetCreator(100, 30, True).getDataset(onehot=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    print(len(dataset))
    print("dataset x:")
    print(dataset[0].x)
    print("dataset edge_index:")
    print(dataset[0].edge_index)
    print("dataset y:")
    print(dataset[0].y)
    y = [data.y for data in dataset]
    classes = np.unique(y)
    train_test_split = 0.8
    train_idx = int(len(dataset)*0.8)
    train_dataset = dataset[:train_idx]
    test_dataset = dataset[train_idx:]
    node_features = len(dataset[0].x[0])
    print("dataset downloaded")
    num_classes = len(classes)
    train_loader = GNNInterpreterLoaderWrapper(train_dataset, batch_size=64, shuffle=True)
    #pre_train_loader = PrefetchLoader(train_loader, device)
    test_loader = GNNInterpreterLoaderWrapper(test_dataset, batch_size=64, shuffle=False)
    #pre_test_loader = PrefetchLoader(test_loader, device)
    print("batches created")

    model = GCNClassifier( node_features=node_features, num_classes=num_classes, hidden_channels = 32).to(device)


    for epoch in trange(32):
        train_loss = fit_model(model, train_loader, lr=0.001)
        train_f1 = evaluate_model(train_loader, model, num_classes=num_classes)
        val_f1 = evaluate_model(test_loader, model, num_classes=num_classes)
        print(f'Epoch: {epoch:03d}, '
            f'Train Loss: {train_loss:.4f}, '
            f'Train F1: {train_f1}, '
            f'Test F1: {val_f1}')

    
    
    torch.save(model.state_dict(), 'model/gnni_model_red_class.pt')
    
    torch.save(train_loader, "model/gnni_test_loader_red_ratio.pt")

    # print("done")
    # model, optimizer = model_optimizer_setup(GCNClassifier, device)
    # for epoch in range(1, 10):
    #     train(model, optimizer, pre_train_loader)
    #     train_acc = test(pre_train_loader, model)
    #     test_acc = test(pre_test_loader, model)
    #     print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # torch.save(model, "model/model_red_ratio.pt")
    # torch.save(train_loader, "model/test_loader_red_ratio.pt")