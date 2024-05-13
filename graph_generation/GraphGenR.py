import torch
from typing import List,Dict
import networkx as nx
from torch_geometric import data
from torch_geometric.data import InMemoryDataset
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot
import random
import uuid

class DatasetCreator:
    def __init__(self, numOfGraphs):
        self.numOfGraphs = numOfGraphs
        self.graphList : List[Graph]= []
    
    def getDataset(self, onehot = False):
        datalist = []
        for g in (self.graphList):
            datalist.append(g.getTorchData(onehot))
        return datalist


class Graph:
    def __init__(self, numOfNodes, connected, param1):
        self.labelVec = []
        self.ID = str(uuid.uuid4())
        if connected:
            self.G = nx.barabasi_albert_graph(numOfNodes,param1)
        else:
            self.G = nx.erdos_renyi_graph(numOfNodes,param1)

    def initNode(self, n, fv, cl):
        self.G.nodes[n]["FV"] = fv
        self.G.nodes[n]["Color"] = cl



    def getNodeFeatureVec(self):
        ## FV is the key where we hold the feature vector of each node
        featureVec = [x[1] for x in self.G.nodes.data("FV")]
        return featureVec


    def getTorchData(self, onehot):
        x = torch.tensor(self.getNodeFeatureVec(), dtype=torch.float)
        if onehot:
            binary_tensor = self.labelVec
            print(binary_tensor)
            # Ensure the tensor is in the correct order, with the least significant bit at the end
            # If your tensor has the LSB at the start, you would use binary_tensor = binary_tensor.flip(dims=[0])
            powers_of_two = torch.pow(2, torch.arange(len(binary_tensor)))
            print(powers_of_two)
            # Multiply each bit by the corresponding power of two and sum up
            y = torch.dot(binary_tensor * powers_of_two)
            
        else:
            y = torch.tensor(self.labelVec)

        ## Since torch does not exactly handles undirected graphs, edge-mirroring is needed for "undirectedness"
        ei = list(self.G.edges())
        for edge in list(self.G.edges()):
            mirrored = edge[::-1]
            ei.append(mirrored)
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        
        return data.Data(x=x, edge_index=edge_index, y=y)            


    def drawG(self, filename):
        ## Color is the key where we hold the desired color of each node
        cm = [x[1] for x in self.G.nodes.data("Color")]
        for c in cm:
            if c == None or c == [0,0,0]:
                cm[cm.index(c)] = "orange"

        fig = matplotlib.pyplot.figure()
        nx.draw(self.G, node_color=cm, node_size=100, edge_color="black", ax=fig.add_subplot())
        matplotlib.use("Agg") 
        fig.savefig(filename)