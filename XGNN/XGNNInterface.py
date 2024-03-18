# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:58:24 2024

@author: hanne
"""

from gnn_explain import gnn_explain
import networkx as nx
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
        
    

class XGNNInterface(gnn_explain):
    """
    This Interface is used to create an instance of the xgnn explain class.
    max_node: maximum number of nodes that should be allowed in the explain graph
    max_step: maximum number of steps in the graph creation (1 step: adding edges(and nodes))
    target_class: specifies the class that should correpsond to the explain graph
    max_iters: number of training iterations
    number_node_types: number of different nodes types
    model: instance of model with same initialization as checkpoint model
    convertNxToData: function that takes an nxgraph and returns an geometric data object
    starting_node: defines starting node in the explain graph, random if None
    roll_out_alpha: hyperparameter for balancing step rewards and future rewards in rollouts 
    checkpoint: path of checkpoint for trained model
    
    To generate a graph, one has to call train on the XGNNInterface instance.
    
    An example can be found in main execution.
    """
    
    
    def __init__(self,max_node, max_step, target_class, max_iters, number_node_types,
                  model, convertNxToData, starting_node =None, roll_out_alpha = 2, 
                  checkpoint = "./checkpoint/ckpt.pth"):
        super(XGNNInterface, self).__init__(max_node, max_step, target_class, max_iters)
        self.target_class = target_class
        #self.criterion = criterion
        #self.optimizer = optimizer
        
        #self.dict = label_dict
        self.gnnNets = model
        self.roll_out_alpha = roll_out_alpha
        
        self.convertNxToData = convertNxToData
        #self.encoding_dict = encoding_dict
        
        self.starting_node = starting_node
        
if __name__ == '__main__':
    import classificationNetwork as cN
    
    encoding_dict = one_hot_encoding(3) #encoding for red, green, blue
    cgd = lambda nx_graph: convertNxToData(nx_graph, encoding_dict) #convering function
    model  = cN.GCN(3,2,25) #model structure
    #graph with 5 nodes, max 10 edges, class 1, 50 training iterations, 3 different node types (red, green ,blue)
    
    explainer = XGNNInterface(5, 10, 1, 50, 3, model = model, convertNxToData = cgd, starting_node=0) 
    graph = explainer.train()
    def printGraph( data): 
        g = torch_geometric.utils.to_networkx(data, to_undirected=True, )
        #nx.draw_networkx(g, with_labels = True)
    
        feature_vector = data.x.numpy()
    
        # Plot the graph with node colors based on the feature vector
        nx.draw_networkx(g, with_labels=True, node_color=feature_vector)
        
    printGraph(graph)