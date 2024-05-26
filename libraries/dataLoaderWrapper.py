import sys
import os

sys.path.append(os.path.abspath(''))
import random
import torch
from torch_geometric.loader import DataLoader
from torchmetrics import F1Score
import torch_geometric as pyg
import networkx as nx

import numpy as np
from torch import nn
from tqdm.auto import tqdm

import utility_functions as utility_functions


class GNNInterpreterLoaderWrapper(DataLoader):
    GRAPH_CLS = {}

    NODE_CLS = {}

    init_done = False
    def initialize(self):
        # check if the dataset.y tensor is a long tensor if it is a float convert it to a long tensor
        for obj in self.dataset:
            if not isinstance(obj.y, torch.LongTensor):
                obj.y = torch.tensor(obj.y, dtype=torch.long)
        self.init_done = True
        max_class = max(obj.y for obj in self.dataset)
        max_node = len(self.dataset[0].x[0])
        self.GRAPH_CLS = {i : str(i) for i in range(max_class +1)}
        identity_matrix = np.eye(max_node)  # Create an n x n identity matrix
        self.NODE_CLS = {i: list(identity_matrix[i]) for i in range(max_node)}


    def __setitem__(self, key, value):
        self.dataset[key] = value

    def __getitem__(self, key):
        return self.dataset[key]

    def __delitem__(self, key):
        del self.dataset[key]

    def __len__(self):
        return len(self.dataset)

    def __getattribute__(self, name):
    # Ensure initialization before accessing NODE_CLS or GRAPH_CLS
        if name in ['NODE_CLS', 'GRAPH_CLS'] and not self.init_done:
            self.initialize()
        return super().__getattribute__(name)
    
    def draw(self, G, pos=None, ax=None):
        labels = [self.NODE_CLS[G.nodes[node]['label']] for node in G.nodes]
        nx.draw(G, pos=pos or nx.kamada_kawai_layout(G), ax=ax, node_color=labels, with_labels=True)

    def set_subplot(self,ax):
        self.ax = ax

    def show(self, idx, ax=None, **kwargs):
        data = self[idx]
        print(f"data: {data}")
        print(f"class: {self.GRAPH_CLS[data.G.graph['label']]}")
        nx.draw_networkx(data.G, with_labels=True, ax=self.ax, node_color= [[] for _ in data.G.nodes['label']])
        
        #self.draw(data.G, ax=ax, **kwargs)

    def describe(self):
        n = [data.G.number_of_nodes() for data in self]
        m = [data.G.number_of_edges() for data in self]
        return dict(mean_n=np.mean(n), mean_m=np.mean(m), std_n=np.std(n), std_m=np.std(m))

    def fit_model(self, model, batch_size=32, lr=0.01):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model.train()
        losses = []
        for batch in self.loader(batch_size=batch_size, shuffle=True):
            model.zero_grad()  # Clear gradients.
            out = model(batch)  # Perform a single forward pass.
            loss = criterion(out['logits'], batch.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            losses.append(loss.item())
        return np.mean(losses)


    @torch.no_grad()
    def evaluate_model(self, model, batch_size=32):
        f1 = F1Score(task="multiclass", num_classes=len(self.GRAPH_CLS), average=None)
        model.eval()
        for batch in self:
            f1(model(batch)['logits'], batch.y)
        return dict(zip(self.GRAPH_CLS.values(), f1.compute().tolist()))

    @torch.no_grad()
    def mean_embeddings(self, model, batch_size=32):
        embeds = [[] for _ in range(len(self.GRAPH_CLS))]
       
        model.eval()
        for batch in DataLoader(self, batch_size=batch_size, shuffle=False):
            
            for i, e in enumerate(model(batch)['embeds']):
                embeds[batch.y[i].item()].append(e)
        return [torch.stack(e, dim=0).mean(axis=0) for e in embeds]
    
    def convert(self, G, generate_label=False):
        if isinstance(G, list):
            return pyg.data.Batch.from_data_list([self.convert(g) for g in G])
        G = nx.convert_node_labels_to_integers(G)
        node_labels = [G.nodes[i]['label']
                       if 'label' in G.nodes[i] or not generate_label
                       else random.choice(list(self.NODE_CLS))
                       for i in G.nodes]
        if G.number_of_edges() > 0:
            if hasattr(self, "EDGE_CLS"):
                edge_labels = [G.edges[e]['label']
                               if 'label' in G.edges[e] or not generate_label
                               else random.choice(list(self.EDGE_CLS))
                               for e in G.edges]
                edge_index, edge_attr = pyg.utils.to_undirected(
                    torch.tensor(list(G.edges)).T,
                    torch.eye(len(self.EDGE_CLS))[edge_labels].float(),
                )
            else:
                edge_index, edge_attr = pyg.utils.to_undirected(
                    torch.tensor(list(G.edges)).T,
                ), None
        else:
            if hasattr(self, "EDGE_CLS"):
                edge_index, edge_attr = torch.empty(2, 0).long(), torch.empty(0, len(self.EDGE_CLS))
            else:
                edge_index, edge_attr = torch.empty(2, 0).long(), None
        return pyg.data.Data(
            G=G,
            x=torch.eye(len(self.NODE_CLS))[node_labels].float(),
            y=torch.tensor(G.graph['label'] if "label" in G.graph else -1).long(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )