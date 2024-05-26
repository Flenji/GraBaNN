import explainers.GNNBoundary.gnn_boundary.models as GBModel

import torch
from tqdm.auto import trange

from torch_geometric.loader import DataLoader
from graph_generation.MultiGraphs import MultiGraphs
import graph_generation.RedRatioGraphs as RedRatioGraphs
import matplotlib.gridspec as gridspec
import graph_generation.HouseSet as HouseSet 

from explainers.gnninterpreter import * 
from torch import nn
import networkx as nx
from torchmetrics import F1Score

import numpy as np
import matplotlib.pyplot as plt

from libraries.dataLoaderWrapper import GNNInterpreterLoaderWrapper

def model_optimizer_setup(model_constr,device):
    
    model = model_constr(node_features=3,  num_classes =2, hidden_channels = 32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return model, optimizer


def train(model, optimizer, loader):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
         out = model(batch= data)  # Perform a single forward pass.
         loss = torch.nn.CrossEntropyLoss()(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader, model):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.loader.dataset)  # Derive ratio of correct predictions.


def fit_model(model, loader, batch_size=32, lr=0.01):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for batch in loader:
        model.zero_grad()  # Clear gradients.
        out = model(batch)  # Perform a single forward pass.
        loss = criterion(out['logits'], batch.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        losses.append(loss.item())
    return np.mean(losses)

def evaluate_model(dataloader,model, num_classes):
    f1 = F1Score(task="multiclass", num_classes=num_classes, average=None)
    model.eval()
    for batch in dataloader:
        f1(model(batch)['logits'], batch.y)
    return dict(zip(range(num_classes), f1.compute().tolist()))

def train_and_save(dataset, device, name):
    
    y = [data.y for data in dataset]
    classes = np.unique(y)
    train_idx = int(len(dataset)*0.8)
    train_dataset = dataset[:train_idx]
    test_dataset = dataset[train_idx:]
    node_features = len(dataset[0].x[0])
   
    print(str(name) + "dataset downloaded")
    num_classes = len(classes)
    train_loader = GNNInterpreterLoaderWrapper(train_dataset, batch_size=64, shuffle=True)
    #pre_train_loader = PrefetchLoader(train_loader, device)
    test_loader = GNNInterpreterLoaderWrapper(test_dataset, batch_size=64, shuffle=False)
    #pre_test_loader = PrefetchLoader(test_loader, device)
    print(str(name)+ "batches created")

    model = GBModel.GCNClassifier( node_features=node_features, num_classes=num_classes, hidden_channels = 32).to(device)


    for epoch in trange(32):
        train_loss = fit_model(model, train_loader, lr=0.001)
        train_f1 = evaluate_model(train_loader, model, num_classes=num_classes)
        val_f1 = evaluate_model(test_loader, model, num_classes=num_classes)
        print(f'Epoch: {epoch:03d}, '
            f'Train Loss: {train_loss:.4f}, '
            f'Train F1: {train_f1}, '
            f'Test F1: {val_f1}')

    
    
    torch.save(model.state_dict(), 'outputs/models/gnni2_model_'+str(name)+ '.pt')
    torch.save(train_loader, 'outputs/models/gnni2_test_loader_'+str(name)+ '.pt')

def generate_and_save_results(model_file, dataset_file, dataset_name = "no_name", temp =0.15):
    datasetLoader = torch.load(dataset_file)
    dataset = datasetLoader.dataset
    y = [data.y for data in dataset]
    classes = np.unique(y)
    node_classes = len(dataset[0].x[0])
    model = GBModel.GCNClassifier(node_features=node_classes,
                        num_classes=len(classes),
                        hidden_channels=32)

    model.load_state_dict(torch.load(model_file))
    print(f'{dataset_name} model loaded')
    mean_embeds = datasetLoader.mean_embeddings(model)
    print(f'{dataset_name} mean embeddings calculated')
    trainer = {}
    sampler ={}

    for c in range(len(classes)):
        trainer[c] = Trainer(
            sampler=(s := GraphSampler(
                max_nodes=40,
                num_node_cls=node_classes,
                temperature=temp,
                learn_node_feat=True,
                
            )),
            discriminator=model,
            criterion=WeightedCriterion([
                dict(key="logits", criterion=ClassScoreCriterion(class_idx=c, mode='maximize'), weight=1),
                dict(key="embeds", criterion=EmbeddingCriterion(target_embedding=mean_embeds[c]), weight=10),
                dict(key="logits", criterion=MeanPenalty(), weight=0),
                dict(key="omega", criterion=NormPenalty(order=1), weight=1),
                dict(key="omega", criterion=NormPenalty(order=2), weight=1),
                dict(key="xi", criterion=NormPenalty(order=1), weight=0),
                dict(key="xi", criterion=NormPenalty(order=2), weight=0),
                # dict(key="eta", criterion=NormPenalty(order=1), weight=0),
                # dict(key="eta", criterion=NormPenalty(order=2), weight=0),
                dict(key="theta_pairs", criterion=KLDivergencePenalty(binary=True), weight=0),
            ]),
            optimizer=(o := torch.optim.SGD(s.parameters(), lr=1)),
            scheduler=torch.optim.lr_scheduler.ExponentialLR(o, gamma=1),
            dataset=datasetLoader,
            budget_penalty=BudgetPenalty(budget=10, order=2, beta=1),
            target_probs={c: (0.9, 1)},
            k_samples=16
        )
        if trainer[c].train(len(dataset)):
            fig, ax = plt.subplots()
            datasetLoader.set_subplot(ax)
            for threshold in [0.5, 0.7, 0.9]:
                graph, show_res = trainer[c].evaluate(threshold=threshold, show=True)
                torch.save(graph, f'outputs/gnni2/{dataset_name}_temp_{temp}_threshold_{threshold}_class_{c}.pt')
                
                formatted_logits = [f"{logit:.4f}" for logit in show_res["logits"]]
                formatted_probs = [f"{prob:.4f}" for prob in show_res["probs"]]
                title = f"n={show_res['n']} m={show_res['m']} logits={formatted_logits} probs={formatted_probs}"
                # Adding dictionary text below the plot
                ax.set_title( title)
                ax.axis('off')
                plt.savefig(f'outputs/gnni2/{dataset_name}_temp_{temp}_threshold_{threshold}_class_{c}.png')
                
            
            print(f'{dataset_name} class {c} temp {temp} done')
        
        else: 
            print(f'{dataset_name} class {c} temp {temp} failed')
            


if __name__=='__main__':
    datasets = []

    # datasets.append({"dataset": RedRatioGraphs.RedRatioGraphs(10000).getDataset(), "name": "RedRatioGraphs"})
    # print('RedRatioGraphs done')
    datasets.append({"dataset":MultiGraphs(10000, negative_class=True).getDataset(), "name": "MultiGraphsTrue"})
    print('MultiGraphsTrue done')
    datasets.append({"dataset":MultiGraphs(10000, negative_class=False).getDataset(), "name": "MultiGraphsFalse"})
    print('MultiGraphsFalse done')
    datasets.append({"dataset":HouseSet.HouseSetCreator(1000, 40,60).getDataset(), "name": "HouseSet"})
    print('HouseSet done')
    

    # for dataset in datasets:
    #     train_and_save(dataset["dataset"], torch.device('cpu'), dataset["name"])
    #     print(f'{dataset["name"]} done')
    
    for dataset in datasets: 
        for temp in [0.1, 0.15, 0.2, 0.4]:
            generate_and_save_results('outputs/models/gnni2_model_'+str(dataset["name"])+'.pt', 'outputs/models/gnni2_test_loader_'+str(dataset["name"])+'.pt', dataset_name=dataset["name"], temp=temp)
            print(f'{dataset["name"]} temp {temp} done')