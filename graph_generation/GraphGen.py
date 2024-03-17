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
        for i in range(self.numOfGraphs):
            self.graphList.append(Graph())
    
    def getDataset(self):
        datalist = []
        return datalist
    
class Graph:

    ## The three defining attributes for graph generation, latter could be used in the future
    def __init__(self, numOfNodes, numOfEdges, directed):
        self.numOfNodes = numOfNodes
        self.numOfEdges = numOfEdges
        self.Nodes : List[Node] = []
        for i in range(numOfNodes):
            self.Nodes.append(Node())
        self.directed = directed
        self.edges = []
        self.labels = []

        ## Generate edges from a pre-defined all-containing pool (random tries can be slow)
        if not numOfNodes == 0: 
            print ("Generating " + str(self.numOfNodes) + " Nodes with " + str(self.numOfEdges) + " Edges.")
            pool = []
            for x in range(self.numOfNodes):
                for y in range(self.numOfNodes):
                    pool.append([x,y])

            i = 0
            while ( i < self.numOfEdges):
                
                nEdge = pool[(random.randint(0, len(pool)-1))]

                self.edges.append(nEdge)
                self.Nodes[nEdge[0]].neighbours.append(self.Nodes[nEdge[1]])
                if not self.directed:
                    self.Nodes[nEdge[1]].neighbours.append(self.Nodes[nEdge[0]])
                i += 1
                pool.remove(nEdge)

                if len(pool) == 0:
                    print("Edge generation pool is empty. Expecting fully connected graph.")
                    break
    
    def getNodeFeatureVec(self):
        featureVec = []
        for node in self.Nodes:
            featureVec.append(node.featureVector)
        return torch.tensor(featureVec, dtype=torch.float)

    def getNodeLabelVec(self):
        y = []
        for node in self.Nodes:
            y.append(node.labelVector)
        return y

    def getTorchData(self, gn):

        ## Since torch does not exactly handles undirected graphs, edge-mirroring is needed for "undirectedness"
        ei = self.edges.copy()
        if not self.directed:
            for edge in self.edges:
                mirrored = edge[::-1]
                ei.append(mirrored)

            ## Multi-edges?
            ##ei = list( dict.fromkeys(ei) )
                
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        
        ## Classification level depends on the dataset generator
        y = []
        if (gn == "graph"):
            y = self.labels
        elif (gn == "node"):
            y = self.getNodeLabelVec
        else:
            print("Use string \"node\" or \"graph\" for the level of classification. Using node level for now.")
            y = self.getNodeLabelVec

        return data.Data(x=self.getNodeFeatureVec(), edge_index=edge_index, y=torch.tensor(y))


    def networkxDraw(self, filename):
        G = nx.Graph()
        color_map = []
        pos = {}
        for node in self.Nodes:
            G.add_node(node)
        for node in self.Nodes:
            for neighbour in node.neighbours:
                G.add_edge(node, neighbour)
        fig = matplotlib.pyplot.figure()
        nx.draw(G, node_size=100, edge_color='black', ax= fig.add_subplot())
    
        matplotlib.use("Agg") 
        fig.savefig(filename)

class Node:
    def __init__(self):
        self.neighbours : List[Node] = []
        self.featureVector = []
        self.labelVector = []