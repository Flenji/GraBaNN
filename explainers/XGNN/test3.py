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
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, os.path.join(parentdir,"graph_generation"))
sys.path.insert(0, parentdir)

import training_network_example as cN
import MultiGraphs as gg
from XGNNInterface import XGNNInterface
import networkx as nx
import torch
import torch_geometric
from utility_functions import *


dummy_gg = gg.MultiGraphs(20)
encoding_dict = one_hot_encoding(3) #encoding for red, green, blue
cgd = lambda nx_graph: convertNxToData(nx_graph, encoding_dict) #convering function
model  = cN.GCN(3,3,25) #model structure

dataset = dummy_gg.getDataset()
#test = dataset[0]

#print(model(test.x, test.edge_index))
#graph with 5 nodes, max 10 edges, class 1, 50 training iterations, 3 different node types (red, green ,blue)

explainer = XGNNInterface(9, 20, 1, 300,3,learning_rate=0.1, model = model,\
                          convertNxToData = cgd, starting_node=1,\
                              checkpoint = "./checkpoint/multi.pth",roll_out_alpha = 2,
                              reward_stepwise = 0.35) 

graph, prob = explainer.train()#

"""
graphs = []
reward_steps = [i*0.1 for i in range(1,11)]
for reward_stepwise in reward_steps:
    explainer = XGNNInterface(10, 12*4,1, 200, 3,learning_rate=0.1, reward_stepwise=reward_stepwise, model = model, convertNxToData = cgd, starting_node=0, checkpoint = "./checkpoint/gg.pth",roll_out_alpha = 1) 
    graph, prob = explainer.train()
    graphs.append((graph,prob))"""
  
    
def printGraph( data): 
    g = torch_geometric.utils.to_networkx(data, to_undirected=True, )
    #nx.draw_networkx(g, with_labels = True)

    feature_vector = data.x.numpy()

    # Plot the graph with node colors based on the feature vector
    nx.draw_networkx(g, with_labels=True, node_color=feature_vector)
    
    
def dataToXandEdgeIndex(data):
    return data.x, data.edge_index

printGraph(graph)




