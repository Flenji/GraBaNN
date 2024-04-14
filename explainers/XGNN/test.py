# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:07:56 2024

@author: hanne
"""
import sys
import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.join(parentdir,"graph_generation"))
sys.path.insert(0, os.path.join(parentdir,"models"))

import grid_graph_gnn as cN
import GridGraphs as gg
from XGNNInterface import XGNNInterface
import networkx as nx
import torch
import torch_geometric


dummy_gg = gg.GridGraphs(20)
encoding_dict = dummy_gg.one_hot_encoding(3) #encoding for red, green, blue
cgd = lambda nx_graph: dummy_gg.convertNxToData(nx_graph, encoding_dict) #convering function
model  = cN.GCN(3,2,25) #model structure

dataset = dummy_gg.getDataset()
#test = dataset[0]

#print(model(test.x, test.edge_index))
#graph with 5 nodes, max 10 edges, class 1, 50 training iterations, 3 different node types (red, green ,blue)


graphs = []
reward_steps = [i*0.1 for i in range(1,11)]
for reward_stepwise in reward_steps:
    explainer = XGNNInterface(10, 12*4,1, 200, 3,learning_rate=0.1, reward_stepwise=reward_stepwise, model = model, convertNxToData = cgd, starting_node=0, checkpoint = "./checkpoint/gg (2).pth",roll_out_alpha = 2) 
    graph, prob = explainer.train()
    graphs.append((graph,prob))
    
    
def printGraph( data): 
    g = torch_geometric.utils.to_networkx(data, to_undirected=True, )
    #nx.draw_networkx(g, with_labels = True)

    feature_vector = data.x.numpy()

    # Plot the graph with node colors based on the feature vector
    nx.draw_networkx(g, with_labels=True, node_color=feature_vector)
    
#printGraph(graph)




