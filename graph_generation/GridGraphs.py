# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:03:12 2024

@author: hanne
"""

import networkx as nx
import random
import torch
import torch_geometric
from numpy.random import choice

class GridGraphs():
    
    def __init__(self,number_graphs):
        #self.max_nodes = max_nodes
        #self.max_edges_per_node = max_edges_per_node
        
        self.data_list = self.generateGraphs(number_graphs)
        
    def getDataset(self):
        return self.data_list
    
    def getPositiveClass(self,total_num_nodes):
        
        rand_nodes = total_num_nodes -9
        rand_graph = nx.barabasi_albert_graph(rand_nodes,2)
        
        node_attr = {}
        number = len(rand_graph)
        for node  in rand_graph.nodes:
            node_attr[node] = {"label": random.randint(0,2)}
        
        
        nx.set_node_attributes(rand_graph, node_attr)
        
        grid_graph = nx.grid_graph((3,3))
        
        
        node_attr = {}
        for index,node in enumerate(grid_graph.nodes):
            node_attr[node] = {"label":index%2}
            
        nx.set_node_attributes(grid_graph, node_attr)
        
        mapping = {node: i+number for i, node in enumerate(grid_graph.nodes())}
        grid_graph = nx.relabel_nodes(grid_graph, mapping)
        
        combined_graph = nx.compose(rand_graph,grid_graph)
        
        attachment_node = random.randint(0,number-1)
        combined_graph.add_edge(attachment_node,number)
        return combined_graph
    
    def getNegativeClass(self, total_num_nodes):
        
        rand_nodes = total_num_nodes-9
        p = [(rand_nodes/3+5)/total_num_nodes, (rand_nodes/3+4)/total_num_nodes,\
             (rand_nodes/3)/total_num_nodes]
        
        rand_graph = nx.barabasi_albert_graph(total_num_nodes,2)
        
        node_attr = {}
        number = len(rand_graph)
        for node  in rand_graph.nodes:
            node_attr[node] = {"label": choice([0,1,2],1,p=p)[0]}
        
        
        nx.set_node_attributes(rand_graph, node_attr)
        return rand_graph
    
    def one_hot_encoding(self,number_node_types):
        """
        This function returns a dictionary that contains keys corresponing to the 
        node_types (number) and values that represent the one hot encoding of the 
        node.
        """
        encoding_dict = {}
        for i in range(number_node_types):
            encoding_dict[i] = [int(j == i) for j in range(number_node_types)]
        return encoding_dict
    
    def convertNxToData(self,nx_graph, encoding_dict):
        """
        This function converts a nx graph to a pytorch geometric data object. For 
        that it converts the label of a node to a one hot incoding feature in the
        data object using the encoding dict.
        """
        x = []
        for node in nx_graph.nodes(data=True):
            try:
                node_type = node[1]["label"]
            except:
                return node
            value = encoding_dict[node_type]
            x.append(value)
        
        if len(list(nx_graph.edges()))!= 0:
            edge_index = torch.tensor(list(nx_graph.edges())).t().contiguous()
        else:
            edge_index = torch.tensor([[],[]], dtype=torch.long)
            
        x = torch.tensor(x,dtype=torch.float)
        
        data = torch_geometric.data.Data(x=x, edge_index=edge_index)
        return data
    
    def generateGraphs(self,number_of_graphs):
        enc_dict = self.one_hot_encoding(3)
        dataset = []
        for i in range(number_of_graphs):
            if i%2 == 0:
                num_nodes = random.randint(30, 40)
                graph = self.getPositiveClass(num_nodes)
                data = self.convertNxToData(graph, enc_dict)
                data.y = torch.Tensor([1])
            else:
                graph = self.getNegativeClass(num_nodes)
                data = self.convertNxToData(graph, enc_dict)
                data.y = torch.Tensor([0])
            dataset.append(data)
        return dataset
    
    
    def printGraph(data): 
        g = torch_geometric.utils.to_networkx(data, to_undirected=True, )
        #nx.draw_networkx(g, with_labels = True)
    
        feature_vector = data.x.numpy()
    
        # Plot the graph with node colors based on the feature vector
        nx.draw_networkx(g, with_labels=True, node_color=feature_vector)
        

    