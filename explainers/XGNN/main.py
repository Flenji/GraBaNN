import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar

from gnn_explain import gnn_explain



#explainer = gnn_explain(3, 30,  0, 50)  ####arguments: (max_node, max_step, target_class, max_iters)

#graph, model1 = explainer.train()



for i in range(1,5+1):
    explainer = gnn_explain(i, 30, 0 , 100)  ####arguments: (max_node, max_step, target_class, max_iters
    graph = explainer.train()



