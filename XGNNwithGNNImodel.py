# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:18:41 2024

@author: hanne
"""

import importlib
import graph_generation.RedRatioGraphs as RedRatioGraphs
import explainers.gnninterpreter.models as models
from torch_geometric.data import Data

from utility_functions import *


import torch


import sys

sys.path.append("./explainers/XGNN")
from XGNNInterface  import XGNNInterface
#from XGNNInterface import encoding_dict
#model = models.GCNClassifier(node_features =3, num_classes =2, hidden_channels=32)

redRatioGraphs = RedRatioGraphs.RedRatioGraphs(100)
dataset = redRatioGraphs.getDataset()

model = models.GCNClassifier(hidden_channels = 32, node_features = 3, num_classes=2)


PATH =  "./model/model_red_class.pt"
model.load_state_dict(torch.load(PATH))

def model_wrapper(x, edge_index):
    data = Data(x=x, edge_index= edge_index)
    forward = model(data)
    return forward["logits"], forward["probs"]

encoding_dict = one_hot_encoding(3)
cgd = lambda nx_graph: convertNxToData(nx_graph, encoding_dict)

explainer = XGNNInterface(5, 15, 1, 100*5, 3, model = model_wrapper, convertNxToData = cgd, starting_node=0,checkpoint=False) 
graph,prob = explainer.train()

printGraph(graph)