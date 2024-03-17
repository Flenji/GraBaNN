import GraphGen
from typing import List,Dict
import random
import networkx as nx
import math

## Should re-write this for interface
class CycleSetCreator (GraphGen.DatasetCreator):
    def __init__(self, numOfGraphs, maxNodes, maxEdges):
        self.numOfGraphs = numOfGraphs
        self.graphList : List[GraphGen.Graph] = []

        ## Generate graphs for datalist
        for i in range(numOfGraphs):
            numOfNodes = random.randint(0, maxNodes)
            numOfEdges = random.randint(0, maxEdges)
            print ("Max Nodes: " + str(maxNodes)+ " --> " + str(numOfNodes))
            print ("Max Edges: " + str(maxEdges)+ " --> " + str(numOfEdges))
            graph = GraphGen.Graph(numOfNodes, numOfEdges, False)
            graph.networkxDraw( "./graphs/cycle/graph" + str(i) )

            print ("Edges: " + str(graph.edges))

            if graph.directed:
                G = nx.DiGraph(graph.edges)
            else:
                G = nx.DiGraph(graph.edges).to_undirected()

            try: 
                nx.find_cycle(G, orientation="original")
                graph.labels = [1]
            except nx.NetworkXNoCycle:
                graph.labels = [0]

            print ("Graph labels: " + str(graph.labels))
            print("")
            
            ## Add graph to list
            self.graphList.append(graph)

    def getDataset(self):
        datalist = []
        for g in (self.graphList):
            datalist.append(g.getTorchData("graph"))
        return datalist
    
## print(CycleSetCreator(3,12,20).getDataset())