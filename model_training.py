import graph_class_gnn

import torch

from graph_generation.MultiGraphs import MultiGraphs
from graph_generation.RedRatioGraphs import RedRatioGraphs
import graph_generation.HouseSet as HouseSet 
import numpy as np

from torch_geometric.loader import DataLoader

from pg_expl_functions import generate_results


def doall(dataset, name):
    device = torch.device('cpu')
    train_test_split = 0.8
    train_idx = int(len(dataset)*0.8)
    print(train_idx)
    train_dataset = dataset[:train_idx]
    test_dataset = dataset[train_idx:]

    print(str(name) + "dataset downloaded")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # pre_train_loader = PrefetchLoader(train_loader, device)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # pre_test_loader = PrefetchLoader(test_loader, device)
    print(str(name)+ "batches created")

    y = [data.y for data in dataset]
    classes = np.unique(y)

    print("done")
    model, optimizer = graph_class_gnn.model_optimizer_setup(graph_class_gnn.Graph_Classification_GCN, device, input_nodes=len(dataset[0].x[0]), output_nodes=len(classes))
    for epoch in range(1, 30):
        graph_class_gnn.train(model, optimizer, train_loader)
        train_acc = graph_class_gnn.test(train_loader, model)
        test_acc = graph_class_gnn.test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    torch.save(model.state_dict(), 'outputs/models/pgexp_model_'+str(name)+ '.pt')
    torch.save(train_loader, 'outputs/models/pgexp_test_loader_'+str(name)+ '.pt')

if __name__ == '__main__':
    datasets = []




    datasets.append({"dataset": RedRatioGraphs.RedRatioGraphs(10000).getDataset(), "name": "RedRatioGraphs"})
    print('RedRatioGraphs done')
    datasets.append({"dataset":MultiGraphs(10000, negative_class=True).getDataset(), "name": "MultiGraphsTrue"})
    print('MultiGraphs done')
    datasets.append({"dataset":MultiGraphs(10000, negative_class=False).getDataset(), "name": "MultiGraphsFalse"})
    print('MultiGraphs done')
    datasets.append({"dataset":HouseSet.HouseSetCreator(1000, 40,60).getDataset(), "name": "HouseSet"})
    print('HouseSet done')
    

   
    for dataset in datasets:
        doall(dataset["dataset"], dataset["name"])
        print(f'{dataset["name"]} done')

    for dataset in datasets: 
        for epoch in [10,25,100]:
            for lr in [0.001, 0.003, 0.005]:
                generate_results('outputs/models/pgexp_model_'+str(dataset["name"])+'.pt', 'outputs/models/pgexp_test_loader_'+str(dataset["name"])+'.pt', epoch=epoch, lr=lr, dataset_name=dataset["name"])
                print(f'{dataset["name"]} epoch {epoch} lr {lr} done')
        
