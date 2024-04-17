# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:33:33 2024

@author: hanne
"""

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

class MultiGraphs():
    
    """
    This class creates random barabasi-albert graphs to which a grit with a 
    specific node coloring is attached (3 different classes).
    If the parameter negative_class is set to true, it will create graphes that
    are divided in 4 classes, where 3 of the classes contain are the grid-graphs 
    and 1 contains barabasi albert graphs without grit attached.
    
    """
    
    def __init__(self,number_graphs, negative_class = False):
        #self.max_nodes = max_nodes
        #self.max_edges_per_node = max_edges_per_node
        
        
        self.red = [0 if index%2 == 0 else 1 for index in range(9)] 
        self.green = [1 if index%2 == 0 else 2 for index in range(9)]
        self.blue = [2 if index%2 == 0 else 0 for index in range(9)] 
        
        self.negative_class = negative_class
        
        self.data_list = self.generateGraphs(number_graphs)
        
        
    
    def createGridShape(self):
        grid_graph = nx.grid_graph((3,3))
        
        
        node_attr = {}
        for index,node in enumerate(grid_graph.nodes):
            node_attr[node] = {"label":index%2}
            
        nx.set_node_attributes(grid_graph, node_attr)
        #mapping = {node: i+number for i, node in enumerate(grid_graph.nodes())}
        #grid_graph = nx.relabel_nodes(grid_graph, mapping)
        return grid_graph, (5,4,0)
    
    def createGridShapeColor(self,colors):
        grid_graph = nx.grid_graph((3,3))
        
        
        node_attr = {}
        for index,node in enumerate(grid_graph.nodes):
            node_attr[node] = {"label":colors[index]}#{"label":index%2}
        
        #print(node_attr)
        nx.set_node_attributes(grid_graph, node_attr)
        
        count = lambda li, val: sum([1 for x in li if x == val])
        #mapping = {node: i+number for i, node in enumerate(grid_graph.nodes())}
        #grid_graph = nx.relabel_nodes(grid_graph, mapping)
        color_dist = (count(colors,0),count(colors,1),count(colors,2))
        return grid_graph, color_dist
        
        
    def createStarShape(self):
        
        star_graph = nx.star_graph(8)
        
        node_attr = {}
        
        for index,node in enumerate(star_graph.nodes):
            if index == 0:
                node_attr[node] = {"label":0}
                continue
            node_attr[node] = {"label":1}
        
        nx.set_node_attributes(star_graph, node_attr)
        
        return star_graph, (1,8,0)
    
        
    def getDataset(self):
        return self.data_list
    
    def getClass(self,total_num_nodes,shape,colors):
        
        
        len_shape = len(shape.nodes)
        rand_nodes = total_num_nodes - len_shape
        
        
        p = [(total_num_nodes/3 - x)/rand_nodes for x in colors]
        
        
        rand_graph = nx.barabasi_albert_graph(rand_nodes,2)
        
        node_attr = {}
        number = len(rand_graph)
        for node  in rand_graph.nodes:
            node_attr[node] = {"label": choice([0,1,2],1,p=p)[0]}
        
        
        nx.set_node_attributes(rand_graph, node_attr)
        
        
        mapping = {node: i+number for i, node in enumerate(shape.nodes())}
        
        shape = nx.relabel_nodes(shape, mapping)
        
        
        combined_graph = nx.compose(rand_graph,shape)
        
        attachment_node = random.randint(0,number-1)
        combined_graph.add_edge(attachment_node,number)
        
        return combined_graph
    
    def getNegativeClass(self,total_num_nodes):
        
        rand_graph = nx.barabasi_albert_graph(total_num_nodes,2)
        node_attr = {}
        number = len(rand_graph)
        for node  in rand_graph.nodes:
            node_attr[node] = {"label": random.randint(0,2)}
        
        
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
        
        #star_graph, star_color = self.createStarShape()
        grid_graph_red, grid_color_red = self.createGridShapeColor(self.red)
        grid_graph_green, grid_color_green = self.createGridShapeColor(self.green)
        grid_graph_blue, grid_color_blue = self.createGridShapeColor(self.blue)
        
        if self.negative_class:
            mod = 4
        else:
            mod = 3
        
        
        for i in range(number_of_graphs):
            if i%mod == 0:
                num_nodes = random.randint(40, 50)
                graph = self.getClass(num_nodes,grid_graph_red, grid_color_red)
                data = self.convertNxToData(graph, enc_dict)
                data.y = torch.Tensor([0])
            elif i%mod == 1:
                graph = self.getClass(num_nodes, grid_graph_green, grid_color_green)
                data = self.convertNxToData(graph, enc_dict)
                data.y = torch.Tensor([1])
            elif i%mod == 2:
                graph = self.getClass(num_nodes, grid_graph_blue, grid_color_blue)
                data = self.convertNxToData(graph, enc_dict)
                data.y = torch.Tensor([2])
            elif i%mod ==3:
                graph = self.getNegativeClass(num_nodes)
                data = self.convertNxToData(graph, enc_dict)
                data.y = torch.Tensor([3])
            
            dataset.append(data)
        return dataset
    
    
    def printGraph(data): 
        g = torch_geometric.utils.to_networkx(data, to_undirected=True, )
        #nx.draw_networkx(g, with_labels = True)
    
        feature_vector = data.x.numpy()
    
        # Plot the graph with node colors based on the feature vector
        nx.draw_networkx(g, with_labels=True, node_color=feature_vector)
        
#x = MultiGraphs(15, negative_class=True)


#dataset = x.getDataset()
 