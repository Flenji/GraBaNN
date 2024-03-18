# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:38:24 2024

@author: hanne
"""

import torch
from torch_geometric.data import Data


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[0,0,1], [0,0,1], [0,0,1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)