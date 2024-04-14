# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:45:10 2024

@author: hanne
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool# global_mean_pool


from torch.nn import Softmax
#from RedRatioGraphs import RedRatioGraphs
import torch.optim as optim
import torch_geometric
import networkx as nx
import os

class GCN(torch.nn.Module):
    def __init__(self,  input_size, num_classes, hidden_channels):
        super(GCN, self).__init__()
        #self.conv1 = GCNConv(input_size, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(input_size, num_classes)
        
        #self.Softmax = Softmax(dim=0)
    
    def forward(self, x, edge_index, batch= None):
        # 1. Obtain node embeddings 
        #x = self.conv1(x, edge_index)
        #x = x.relu()
        #x = self.conv2(x, edge_index)
        #x = x.relu()
        #x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_add_pool(x,batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        logits = self.lin(x)#.squeeze()
        probs = F.softmax(logits, dim=-1)
        #probs = self.Softmax(logits)#, dim = 0)
        #print(logits)
        #print(probs)
        return logits ,probs
    


if __name__ == '__main__':
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, os.path.join(parentdir,"graph_generation"))
    #from RedRatioGraphs import RedRatioGraphs
    from NodeCountGraphs import NodeCountGraphs
    nodeCountGraphs = NodeCountGraphs(1000)
    dataset = nodeCountGraphs.getDataset()
    
    feature_size = len(dataset[0].x[0])
    
    num_classes = 2
    
    model = GCN(feature_size, num_classes, hidden_channels=25)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)#, momentum=0.9, weight_decay=5e-4)
    model.train()
    
    print('The total num of dataset is ', len(dataset))
    best_acc = 0
    smallest_loss = 1000
    for epoch in range(100):
        acc = []
        loss_list = []
        for data in dataset:
            optimizer.zero_grad()
            logits, probs = model(data.x,data.edge_index)
            prediction = logits.argmax(dim=1)
            loss = criterion(logits, data.y.type(torch.LongTensor))
            
            loss_list.append(loss.item())
            loss.backward() #backpropagation
            optimizer.step() #gradient
            
            acc.append(prediction.eq(data.y).item())
        print("Epoch:%d  |Loss: %.3f | Acc: %.3f"%(epoch, np.average(loss_list), np.average(acc)))
        if(np.average(loss_list)<smallest_loss):
            print('saving....')
            state = {
                 'net': model.state_dict(),
                 'acc': np.average(acc),
                 'epoch': epoch,
             }        
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/nodeCount.pth')
            smallest_loss = np.average(loss_list)
            #best_acc = np.average(acc) 
    
    #redRatioGraphs.printGraph(dataset[0])
    
    def test(data):
        logits, probs = model(data.x, data.edge_index,)
        pred = logits.argmax(dim=1)
        
        print(f"predicted class { pred.item()}\nactual class { data.y.item()}")
        
    g = torch_geometric.utils.to_networkx(dataset[0], to_undirected=True,  )
    
    def convertDataToNx(data):
        G = nx.Graph()
        
        for i, features in enumerate(data.x):
            match list(features.numpy()):
                case [1,0,0]:
                    node_type = "r"
                case [0,1,0]:
                    node_type = "g"
                case [0,0,1]:
                    node_type = "b"
            G.add_node(i, label = node_type)
            
        for src, dst in data.edge_index.t().tolist():
            G.add_edge(src,dst)
            
        label = data.y.item()
        return G
            