# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:57:27 2024

@author: hanne
"""

import networkx as nx
import random
import torch
import torch_geometric


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

class NodeCountGraphs():
    
    def __init__(self,number_graphs):
        #self.max_nodes = max_nodes
        #self.max_edges_per_node = max_edges_per_node
        
        self.data_list = self.generateGraphs(number_graphs)

    def getGraph(self):
        
        rand_int = random.randint(5, 15)
        rand_graph = nx.barabasi_albert_graph(rand_int,2)
        
        node_attr = {}
        number = len(rand_graph)
        for node  in rand_graph.nodes:
            node_attr[node] = {"label": 0}
        nx.set_node_attributes(rand_graph, node_attr)
        return rand_graph
    
    def getDataset(self):
        return self.data_list
    
    
    def generateGraphs(self,number_of_graphs):
        enc_dict = one_hot_encoding(1)
        dataset = []
        for i in range(number_of_graphs):
            nx_graph = self.getGraph()
            data = convertNxToData(nx_graph, enc_dict)
            data.y = torch.Tensor([0]) if len(data.x) < 11 else torch.Tensor([1])
            dataset.append(data)
        return dataset
        






