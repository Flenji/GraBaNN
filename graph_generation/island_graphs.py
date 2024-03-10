import math
import torch
from typing import List,Dict
import networkx as nx
from torch_geometric import data
from torch_geometric.data import InMemoryDataset
import heapq


import matplotlib
import matplotlib.pyplot
import random

class DatasetCreator:
    def __init__(self, numOfGraphs, sizeFrom, sizeTo):
        self.numOfGraphs = numOfGraphs
        self.permutationsPerGraph = math.ceil( numOfGraphs / (sizeTo - sizeFrom))
        self.graphList : List[Graph2D]= []
        for i in range(sizeFrom, sizeTo):
            graph = Graph2D(i*i)
            graph.setUpNodeNeighbours()
            self.graphList.append(graph)
    
    def getDataset(self):
        datalist = []
        for i in range(self.permutationsPerGraph):
            for graph in self.graphList:
                graph.resetAllColors()
                graph.colorGraph()
                datalist.append(graph.getTorchData())
        
        return datalist


class Graph2D:
    def __init__(self, numOfNodes):
        self.nodes: List[Node] = []
        self.nodeDict: Dict[int, Node] = {}
        self.width = math.ceil(math.sqrt(numOfNodes))

        for i in range(self.width):
            for j in range(self.width):
                node = Node(i*self.width+j,[i,j])
                self.nodes.append(node)
                self.nodeDict[(i,j)] = node
    
    def setUpNodeNeighbours(self):
        for node in self.nodes:
            x = node.placeInfo[0]
            y = node.placeInfo[1]
            if (x, y -1) in self.nodeDict:
                node.addNeighbour(self.nodeDict[(x, y -1)])
            if (x, y +1) in self.nodeDict:
                node.addNeighbour(self.nodeDict[(x, y +1)])
            if (x + 1, y) in self.nodeDict:
                node.addNeighbour(self.nodeDict[(x + 1, y)])
            if (x - 1, y) in self.nodeDict:
                node.addNeighbour(self.nodeDict[(x - 1, y)])
    
    def getEdgeIndex(self):
        edgeIndex = [[],[]]
        for node in self.nodes:
            for neighbour in node.neighbourDict.keys():
                edgeIndex[0].append(node.index)
                edgeIndex[1].append(neighbour)
        return torch.tensor(edgeIndex, dtype=torch.long)
    
    def getNodeFeatureVec(self):
        featureVec = []
        for node in self.nodes:
            featureVec.append(node.getFeatureVec())
        return torch.tensor(featureVec, dtype=torch.float)



    def getTorchData(self):
        return data.Data(x=self.getNodeFeatureVec(), edge_index=self.getEdgeIndex(),y=torch.tensor(self.Y, dtype=torch.long))
    
    def networkxDraw(self, filename):
        G = nx.Graph()
        color_map = []
        pos = {}
        for node in self.nodes:
            G.add_node(node.index)
            color_map.append(node.color)
            pos[node.index] = (node.placeInfo[0], node.placeInfo[1])
        for node in self.nodes:
            for neighbour in node.neighbourDict.keys():
                G.add_edge(node.index, neighbour)
        fig = matplotlib.pyplot.figure()
        nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=100, edge_color='gray', ax= fig.add_subplot())
    
        matplotlib.use("Agg") 
        fig.savefig(filename)

    def resetAllColors(self):
        for node in self.nodes:
            node.color = "red"


    def colorNodesWithDistance(self, dist, color, centernode):
        centerx = centernode.placeInfo[0]
        centery = centernode.placeInfo[1]
        coordinates = get_points_with_manhattan_distance(centerx, centery, distance=dist)
        for coord in coordinates:
            if coord in self.nodeDict:
                self.nodeDict[coord].color = color
    
    def colorGraph(self):
        first_width = int(self.width/4)
        second_width = int(self.width/4)

        self.centernode1 = random.choice(self.nodes)
        self.colorNodesWithDistance(first_width, "blue", self.centernode1)
        
        self.centernode2 = random.choice(self.nodes)
        self.colorNodesWithDistance(second_width, "blue", self.centernode2)

        if heuristic(self.centernode1, self.centernode2) <= first_width + second_width +1:
            self.Y = True
        else:
            self.Y = False


            

        
def heuristic(node, end):
    # Compute Manhattan distance as heuristic
    return abs(node.placeInfo[0] - end.placeInfo[0]) + abs(node.placeInfo[1] - end.placeInfo[1])


class Node:
    def __init__(self, idx, placementInfo):
        self.index = idx
        #x and y coordinates
        self.placeInfo = placementInfo
        self.neighbourDict :Dict[int,Node] = {}
        self.color = "red"
    
    def addNeighbour(self, node):
        self.neighbourDict[node.index] = node

    def getFeatureVec(self):
        if self.color == "red":
            return [1,0]
        if self.color == "blue":
            return [0,1]

    
    def getY(self):
        if self.color == "lightgray":
            return 0
        if self.color == "red":
            return 1
        if self.color == "blue":
            return 2
        if self.color == "green":
            return 3
        if self.color == "yellow":
            return 4
        if self.color == "purple":
            return 5    

def get_points_with_manhattan_distance(x, y, distance):
    points = []
    for i in range(-distance, distance + 1):
        for j in range(-distance, distance + 1):
            if abs(i) + abs(j) <= distance:
                points.append((x + i, y + j))
    return points



if __name__=='__main__':
    DatasetCreator = DatasetCreator(100, 16, 32)
    dataset = DatasetCreator.getDataset()
    
    DatasetCreator.graphList[0].networkxDraw("graph1.png")
    print(DatasetCreator.graphList[0].Y)
    DatasetCreator.graphList[1].networkxDraw("graph2.png")
    print(DatasetCreator.graphList[1].Y)
    
    DatasetCreator.graphList[2].networkxDraw("graph3.png")
    print(DatasetCreator.graphList[2].Y)
    
    DatasetCreator.graphList[3].networkxDraw("graph4.png")
    print(DatasetCreator.graphList[3].Y)
    print(len(dataset))
    #print the number of true and false in the Y vector
    true = 0
    false = 0
    for data in dataset:
        
            if data.y == 1:
                true += 1
            else:
                false += 1
    print(true, false)