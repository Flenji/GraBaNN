import torch
from typing import List,Dict
import networkx as nx
from torch_geometric import data
from torch_geometric.data import InMemoryDataset

import matplotlib
import matplotlib.pyplot
import random

class DatasetCreator:
    def __init__(self, numOfGraphs):
        self.numOfGraphs = numOfGraphs
        self.graphList : List[Graph]= []
    
    def getDataset(self):
        datalist = []
        for g in (self.graphList):
            datalist.append(g.getTorchData())
        return datalist


class Graph:
    def __init__(self, numOfNodes, connected):
        self.labelVec = []
        if connected:
            print(numOfNodes)
            self.G = nx.barabasi_albert_graph(numOfNodes,1)
        else:
            self.G = nx.erdos_renyi_graph(numOfNodes,0.1)

    def initNode(self, n, fv, cl):
        self.G.nodes[n]["FV"] = fv
        self.G.nodes[n]["Color"] = cl



    def getNodeFeatureVec(self):
        ## FV is the key where we hold the feature vector of each node
        featureVec = [x[1] for x in self.G.nodes.data("FV")]
        return featureVec


    def getTorchData(self, G):
        x = torch.tensor(self.getNodeFeatureVec(), dtype=torch.float)
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