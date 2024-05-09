# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:45:41 2024

@author: hanne
"""


import networkx as nx
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.data import Data

def one_hot_encoding(number_node_types):
    """
    This function returns a dictionary that contains keys corresponing to the 
    node_types (number) and values that represent the one hot encoding of the 
    node.
    """
    encoding_dict = {}
    for i in range(number_node_types):
        encoding_dict[i] = [int(j == i) for j in range(number_node_types)]
    return encoding_dict

def convertNxToData(nx_graph, encoding_dict):
    """
    This function converts a nx graph to a pytorch geometric data object. For 
    that it converts the label of a node to a one hot incoding feature in the
    data object using the encoding dict.
    """
    x = []
    for node in nx_graph.nodes(data=True):
        node_type = node[1]["label"]
        value = encoding_dict[node_type]
        x.append(value)
    
    if len(list(nx_graph.edges()))!= 0:
        edge_index = torch.tensor(list(nx_graph.edges())).t().contiguous()
    else:
        edge_index = torch.tensor([[],[]], dtype=torch.long)
        
    x = torch.tensor(x,dtype=torch.float)
    
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    return data

def printGraph(data, filename=""): 
    g = torch_geometric.utils.to_networkx(data, to_undirected=True, )
    #nx.draw_networkx(g, with_labels = True)

    feature_vector = data.x.numpy()

    # Plot the graph with node colors based on the feature vector
    nx.draw_networkx(g, with_labels=False, node_color=feature_vector)
    plt.box(False)
    
    if filename:
        # Save the graph as a PNG file
        plt.savefig(filename)

def printGraphFromNX(data, nxData):
 
    feature_vector = data.x.numpy()
    nx.draw_networkx(nxData, with_labels=True, node_color=feature_vector)
    
    
def model_wrapper(x, edge_index, model):
    """
    Function to match output of GNNInterpreter models to XGNN models.
    """
    data = Data(x=x, edge_index= edge_index)
    forward = model(data)
    return forward["logits"], forward["probs"]

