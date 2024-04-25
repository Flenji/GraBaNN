import torch_geometric.explain as ex
import torch
from graph_class_gnn import Graph_Classification_GCN

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import utility_functions as uf

def generate_results(model_file, dataset_file, epoch = 30, lr = 0.003, example_per_class = 5, dataset_name= "no_name"):
    datasetLoader = torch.load(dataset_file)
    
    dataset = datasetLoader.dataset
    y = [data.y for data in dataset]
    classes = np.unique(y)
    model = Graph_Classification_GCN(input_nodes=len(dataset[0].x[0]), output_nodes=len(classes))
    model.load_state_dict(torch.load(model_file))
    explainer = ex.Explainer(
        model=model,
        algorithm=ex.PGExplainer(epochs=epoch, lr=lr),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=ex.ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )
    
    for epoch in range(epoch):
        #print epochs
        print('Epoch:', epoch)
        for i in range(len(dataset)):
            explainer.algorithm.train(model=model,x=dataset[i].x, edge_index=dataset[i].edge_index,target=dataset[i].y, epoch=epoch)
    print('Training done')
    
    y = [data.y for data in dataset]
    classes = np.unique(y)
    class_indexes = []
    torch.save(explainer, 'outputs/pgexp/pgexp_explainer_epoch_'+str(epoch)+'_lr_'+str(lr)+'_'+str(dataset_name)+'.pt')
    for c in classes:
        class_indexes.append(np.where(y == c)[0])
    for example in range(example_per_class):    
        for c in classes: 
            index = class_indexes[c][example]
            explanation = explainer(x=dataset[index].x, edge_index=dataset[index].edge_index, target=dataset[index].y)
            torch.save(explanation, 'outputs/pgexp/pgexp_explanation_'+str(epoch)+'_lr_'+str(lr)+'_'+str(dataset_name)+'_'+str(c)+'_'+str(example)+'.pt')
            
            uf.printGraph(explanation)
            plt.savefig('outputs/pgexp/pgexp_explanation_'+str(epoch)+'_lr_'+str(lr)+'_'+str(dataset_name)+'_'+str(c)+'_'+str(example)+'_fig.png')

