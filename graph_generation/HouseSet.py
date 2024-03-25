import GraphGen
from typing import List,Dict
import random
import uuid
import networkx as nx
import math

## Should re-write this for interface
class HouseSetCreator (GraphGen.DatasetCreator):
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

            ## Appr. 25% positive cases (+ random generated ones)
            if (random.randint(0,4) == 0):
                self.buildHouse(graph)

            if graph.directed:
                G = nx.DiGraph(graph.edges)
            else:
                G = nx.DiGraph(graph.edges).to_undirected()

            if self.hasHouse(graph):
                graph.labels = [1]
            else:
                graph.labels = [0]
            
            print ("Edges: " + str(graph.edges))

            print ("Node feature vectors: " + (str(graph.getNodeFeatureVec())))
            
            self.paintHouse(graph)

            print ("Graph labels: " + str(graph.labels))
            print("")
            
            graph.networkxDraw( "./graphs/house/graph-" + str(uuid.uuid4()) +".jpg" )
            
            ## Add graph to list
            self.graphList.append(graph)

    def hasHouse(self, graph):
        for m in (graph.Nodes):
            m.featureVector = [0]

        for n in (graph.Nodes):
            for i in (n.neighbours):
                for ii in (i.neighbours):
                    for iii in (ii.neighbours):
                        for v in (iii.neighbours):
                            if v == n:
                                flag = len(set([i,ii,iii,v])) == len([i,ii,iii,v])
                                if flag:
                                    print ("House-base generated: " + str([i,ii,iii,v]))
                                    if self.hasHouseRoof([i,ii,iii,v]):
                                        return True
        return False
    
    def hasHouseRoof(self, hb):
        for n in (hb):
            for i in (n.neighbours):
                if not (i in hb):
                    for ii in (i.neighbours):
                        if not (ii == n) and (ii in hb):
                            print ("Roof found: " + str([n,i,ii]))
                            for walls in hb:
                                walls.featureVector = [1]
                            for roof in [n,i,ii]:
                                roof.featureVector = [2]
                            return True
        print ("But valid roof not found!")
        return False
    
    def buildHouse(self,graph):
        id = len(graph.Nodes)
        w1 = GraphGen.Node()
        w2 = GraphGen.Node()
        r1 = GraphGen.Node()
        r2 = GraphGen.Node()
        r3 = GraphGen.Node()

        house = [w1,w2,r1,r2,r3]

        w1.neighbours.append(w2)
        graph.edges.append([id, id+1])
        w1.neighbours.append(r1)
        graph.edges.append([id, id+2])

        w2.neighbours.append(w1)
        graph.edges.append([id+1, id])
        w2.neighbours.append(r2)
        graph.edges.append([id+1, id+3])

        r1.neighbours.append(r2)
        graph.edges.append([id+2, id+3])
        r1.neighbours.append(w1)
        graph.edges.append([id+2, id])
        r1.neighbours.append(r3)
        graph.edges.append([id+2, id+4])

        r2.neighbours.append(r1)
        graph.edges.append([id+3, id+2])
        r2.neighbours.append(w2)
        graph.edges.append([id+3, id+1])
        r2.neighbours.append(r3)
        graph.edges.append([id+3, id+4])

        r3.neighbours.append(r1)
        graph.edges.append([id+4, id+2])
        r3.neighbours.append(r2)
        graph.edges.append([id+4, id+3])

        graph.Nodes.append(w1)
        graph.Nodes.append(w2)
        graph.Nodes.append(r1)
        graph.Nodes.append(r2)
        graph.Nodes.append(r3)

        ra = random.randint(0, id)
        graph.edges.append([ra,id+4])
        graph.edges.append([id+4,ra])
        r3.neighbours.append(graph.Nodes[ra])
        graph.Nodes[ra].neighbours.append(r3)

        ra = random.randint(0, id)
        graph.edges.append([ra,id])
        graph.edges.append([id,ra])
        w1.neighbours.append(graph.Nodes[ra])
        graph.Nodes[ra].neighbours.append(w1)

        ra = random.randint(0, id)
        graph.edges.append([ra,id+1])
        graph.edges.append([id+1,ra])
        w2.neighbours.append(graph.Nodes[ra])
        graph.Nodes[ra].neighbours.append(w2)
        

    def paintHouse(self, graph):
        cm = []
        for n in graph.getNodeFeatureVec():
            if n == [0]:
                cm.append("black")
            elif n == [1]:
                cm.append("yellow")
            elif n == [2]:
                cm.append("red")
            elif n == []:
                cm.append("blue")

        graph.color_map = cm

    def getDataset(self):
        datalist = []
        for g in (self.graphList):
            datalist.append(g.getTorchData("graph"))
        return datalist

##for i in range(10):
##    rNodes = random.randint(0, 20)
##    rEdges = random.randint(0, rNodes)
##    print(HouseSetCreator(1,rNodes,rEdges).getDataset())