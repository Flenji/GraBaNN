# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:18:41 2024

@author: hanne
"""

import importlib

import explainers.gnninterpreter.models as models
from torch_geometric.data import Data

from utility_functions import *


import torch


import sys

sys.path.append("./explainers/XGNN")
sys.path.append("./graph_generation")
import HouseSet as HouseSet


from XGNNInterface  import XGNNInterface
#from XGNNInterface import encoding_dict
#model = models.GCNClassifier(node_features =3, num_classes =2, hidden_channels=32)

dataset = HouseSet.HouseSetCreator(1000, 40, 60).getDataset()

model = models.GCNClassifier(hidden_channels = 32, node_features = 3, num_classes=2)


PATH =  "./model/house_gnn_interpreter_model.pt"
model.load_state_dict(torch.load(PATH))

def model_wrapper(x, edge_index):
    data = Data(x=x, edge_index= edge_index)
    forward = model(data)
    return forward["logits"], forward["probs"]

encoding_dict = one_hot_encoding(3)
cgd = lambda nx_graph: convertNxToData(nx_graph, encoding_dict)



def diffHyperparameters(starting_node, cl):
    reward_stepwise_li = [x* 0.2 for x in range(1,11-5)]
    roll_out_alpha_li = [0.5,1,3]
    
    res_li = []
    for reward_stepwise in reward_stepwise_li:
        for roll_out_alpha in roll_out_alpha_li:
            explainer = XGNNInterface(5, 20, cl, 200, 3, model = model_wrapper,\
                                      convertNxToData = cgd, starting_node=starting_node,\
                                          checkpoint=False, reward_stepwise=reward_stepwise,\
                                              roll_out_alpha = roll_out_alpha) 
            graph,prob = explainer.train()
            print(f"Starting Node {starting_node}\n rew_step: {reward_stepwise} roll_out_alpha {roll_out_alpha}: prob {prob}")
            res_li.append((graph,prob))
    return max(res_li, key=lambda x:x[1])

res_li = []
for starting_node in [0,1,2]:
    graph, prob = diffHyperparameters(starting_node, 1)
    res_li.append((graph,prob))
    printGraph(graph)

