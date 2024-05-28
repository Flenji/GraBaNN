import networkx as nx
import pandas as pd
import torch_geometric as pyg
#import MultiGraphs

#from base_graph_dataset import BaseGraphDataset
#from utils import default_ax, unpack_G
from gnn_boundary.datasets import *
import torch
import torch_geometric


#dataset = MultiGraphs.MultiGraphs(1000).getDataset()





def one_hot_encoding(number_node_types):
    """
    This function returns a dictionary that contains keys corresponing to the 
    node_types (number) and values that represent the one hot encoding of the 
    node.
    """
    encoding_dict = {}
    for i in range(number_node_types):
        encoding_dict[i] = tuple([int(j == i) for j in range(number_node_types)])
    return encoding_dict

def convertDataToNx(data, index = -1):
    node_types = len(data.x[0])
    G = nx.Graph()
    
    
    dd = one_hot_encoding(node_types)
    dd_inv = {v:k for k,v in dd.items()}
    
    for i in range(data.num_nodes):
        attr = tuple([item.item() for item in data["x"][i]])
        node_attr = {"label": dd_inv[attr]}
        G.add_node(i, **node_attr)
        
    edge_index = data.edge_index.cpu().numpy()
    for j in range(edge_index.shape[1]):
        src, dst = edge_index[:,j]
        G.add_edge(src.item(), dst.item())
    G.graph["index"] = index
    return G

def convertNxToData(nx_graph, encoding_dict):
    """
    This function converts a nx graph to a pytorch geometric data object. For 
    that it converts the label of a node to a one hot incoding feature in the
    data object using the encoding dict.
    """
    mapping = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    
    
    
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
    


class DatasetWrapper(BaseGraphDataset):


    def __init__(self, dataset,num_cls = 2,node_feat = 3 , name="datasetWrapper", color_dict = {0:"red",1:"green",2:"blue"},**kwargs):
        self.dataset = dataset
        
        
        self.GRAPH_CLS = {i:i for i in range(num_cls)}
        self.encoding_dict = one_hot_encoding(node_feat)
        self.convertNxToData = lambda nx_graph:convertNxToData(nx_graph, self.encoding_dict)
        self.color_dict = color_dict
        
        
        super().__init__(name=name)
        
    #def download(self):
    #    super().download()
        
    def generate(self):
        for index, data in enumerate(self.dataset):
            G = convertDataToNx(data, index)
            yield G
    
    def convert(self, G, generate_label =False):
        if generate_label == False:
            index = G.graph["index"]
            data = self.dataset[index]
            data.G = G
        elif generate_label:
            data = self.convertNxToData(G)
            data.G = G
        return data
    
    def draw(self, G, pos= None, ax= None):
        node_colors = []
        for node in G.nodes():
            label = G.nodes[node]["label"]
            node_colors.append(self.color_dict[label])
        
        nx.draw(G,node_color=node_colors)
    
    def process(self):
        super().process()
        
    #@property
    #def raw_file_names(self):
    #    return ["test.txt"]
        
#dataset_ = DatasetWrapper(dataset)    
    