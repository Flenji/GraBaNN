
import GraphGenR

from typing import List,Dict
import random
import uuid
import networkx as nx
import math

## Labeling:
##      [0,0,0] -> negative class (neither blob has a uniform coloring)
##      [0,0,1] -> red blob present
##      [0,1,0] -> blue blob present
##      [1,0,0] -> black blob present
##      Or could be two at the same time
##
## Feature Vectors:
##      [0,0,1] -> red node
##      [0,1,0] -> blue node
##      [1,0,0] -> black node

class DuoSetCreator (GraphGenR.DatasetCreator):
    def __init__(self, numOfGraphs, maxNodes, negativeClass):
        self.numOfGraphs = numOfGraphs
        self.graphList : List[GraphGenR.Graph] = []
        
        ## Generate graphs for datalist
        for i in range(numOfGraphs):
            
            ## Todo because Barabásinak nem tetszik az egyes
            numOfNodes1 = random.randint(2, maxNodes-2)
            numOfNodes2 = (maxNodes-numOfNodes1)

            sG1 = GraphGenR.Graph(numOfNodes1, True)
            sG2 = GraphGenR.Graph(numOfNodes2, True)
            UG = GraphGenR.Graph(2,False)

            ## This should be the higher scope, but it would look shit, TODo
            if negativeClass:

                ## 50-50 if positive or negative class
                positive = random.randint(0, 1)
                if positive == 1:
                    UG.labelVec = self.genPos(sG1,sG2)

                else:
                   UG.labelVec = self.genNeg(sG1,sG2)


            ## Since negative classes are toggleable;
            else:
                UG.labelVec = self.genPos(sG1,sG2)

            
            UG.G = nx.disjoint_union(sG1.G, sG2.G)
            UG.drawG("./graphs/test/graph-" + "test" +".jpg")
            self.graphList.append(UG)


    def bogus(self, subGraph):
        colors = ["red","blue","black"]
        for n in range(len(subGraph.G.nodes())):
            R = random.choice(colors)
            subGraph.initNode(n,self.c2f(R),R)
        
        for n in range(len(subGraph.G.nodes())):
            if subGraph.G.nodes[n]["Color"] != subGraph.G.nodes[0]["Color"]:
                return
        self.bogus(subGraph)

    
    def colorU(self, subGraph, color):
        for n in range(len(subGraph.G.nodes())):
            subGraph.initNode(n,self.c2f(color),color)

    
    def c2f(self,color):
        if color == "red":
            return [0,0,1]
        elif color == "blue":
            return [0,1,0]
        elif color == "black":
            return [1,0,0]
        else:
            return [0,0,0]


    def messUp(self, subGraph):
        colors = ["red","blue","black"]
        R1 = random.choice(colors)
        colors.remove(R1)
        R2 = random.choice(colors)

        self.colorU(subGraph, R1)
        n = random.randint(0, len(subGraph.G.nodes())-1)
        subGraph.initNode(n,self.c2f(R2),R2)

    def genPos(self,sG1, sG2):

        colors = ["red","blue","black"]
        rc1 = random.choice(colors)
        rc2 = random.choice(colors)

        ## One positive is needed for sure
        self.colorU(sG1,rc1)
        ret = self.c2f(rc1)

        ## Second blob is random
        r = random.randint(0,9)
        if r == 0:
            self.messUp(sG2)
        elif r < 5:
            self.colorU(sG2,rc2)
            if rc2 == "red":
                ret[2] = 1
            elif rc2 == "blue":
                ret[1] = 1
            elif rc2 == "black":
                ret[0] = 1
        else:
            self.bogus(sG2)

        return ret

    def genNeg(self,sG1,sG2):
        r1 = random.randint(0,9)
        r2 = random.randint(0,9)

        if r1 == 0:
            self.messUp(sG1)
        else:
            self.bogus(sG1)

        if r2 == 0:
            self.messUp(sG2)
        else:
            self.bogus(sG2)
        
        return [0,0,0]