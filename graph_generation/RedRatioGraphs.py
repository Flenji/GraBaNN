import torch
from torch_geometric.data import Data
import numpy as np
import itertools
import networkx as nx
import torch_geometric



class RedRatioGraphs():
    
    def __init__(self,number_graphs,max_nodes =10, max_edges_per_node = 5):
        self.max_nodes = max_nodes
        self.max_edges_per_node = max_edges_per_node
        
        self.data_list = self.generateGraphs(number_graphs)
        
    def getDataset(self):
        return self.data_list
    
    def generateGraphs(self, number_graphs):
        res = []
        for index in range(number_graphs):
            data = self.generateGraph(self.max_nodes, self.max_edges_per_node)
            data.y = torch.Tensor([self.labelGraph(data)])
            res.append(data)
        return res 
        
    
    
    def generateGraph(self,max_nodes = 10, max_edges_per_node= 5):
        nodes_number = np.random.randint(1,max_nodes)
        nodes = np.arange(nodes_number)
        
        edge_index = []
        for node in nodes:
            number_edges = np.random.randint(1,max_edges_per_node)
            for i in range(number_edges):
                neighbour_node = np.random.choice(nodes)
                edge_index.append([node,np.random.choice(nodes)])
        
        edge_index.sort()
        edge_index = list(k for k,_ in itertools.groupby(edge_index))
        features = []
        for node in nodes:
            decision = np.random.randint(3)
            match decision:
                case 0:
                    x = [1,0,0]
                case 1:
                    x = [0,1,0]
                case 2:
                    x = [0,0,1]
            features.append(x)
            
        
        data = Data(x=torch.Tensor(features), edge_index=torch.tensor(edge_index).t().contiguous())
        return data
    
    def labelGraph(self,data):
        features = data.x.numpy()
        red_counts = 0
        for feature in features:
            if ([1,0,0] == feature).all():
                red_counts += 1
        
        if red_counts/len(features) >= 0.5:
            return 1
        else:
            return 0
    
    def printGraph(data): 
        g = torch_geometric.utils.to_networkx(data, to_undirected=True, )
        #nx.draw_networkx(g, with_labels = True)
    
        feature_vector = data.x.numpy()
    
        # Plot the graph with node colors based on the feature vector
        nx.draw_networkx(g, with_labels=True, node_color=feature_vector)