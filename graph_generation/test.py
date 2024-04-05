import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import HouseSet


### Defining the dataset (graph)

edge_index = torch.tensor([[0, 2],
                           [0, 12],
                           [0, 14],
                           [2, 1],
                           [2, 4],
                           [2, 0],
                           [1, 2],
                           [4, 2],
                           [4, 3],
                           [4, 5],
                           [4, 8],
                           [4, 9],
                           [3, 4],
                           [5, 4],
                           [5, 6],
                           [6, 5],
                           [6, 7],
                           [6, 8],
                           [7, 6],
                           [8, 6],
                           [8, 4],
                           [9, 4],
                           [9, 10],
                           [10, 11],
                           [10, 12],
                           [11, 10],
                           [12, 0],
                           [12, 13],
                           [12, 14],
                           [13, 12],
                           [14, 0],
                           [14, 12]], dtype=torch.long)
x = torch.tensor([[0],
                [-1],
                [1],
                [0],
                [0],
                [1],
                [1],
                [0],
                [-1],
                [0],
                [0],
                [-1],
                [0],
                [1],
                [-1]
                ], dtype=torch.float)

train_mask = torch.tensor([True,True,True,False,False,False,True,True,True,True,True,False,True,True,True])
test_mask = torch.tensor([False,False,False,True,True,True,False,False,False,False,False,True,False,False,False])
labels = torch.tensor([1,0,1,1,1,1,1,1,1,1,0,0,1,0,1])

##dataset = Data(x=x, edge_index=edge_index.t().contiguous(), train_mask=train_mask, test_mask=test_mask, y=labels)
dataset = HouseSet.HouseSetCreator(10000,16,32).getDataset()


### Graph Convolutional Network model

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.conv4 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

       
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


### Hardware setup

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
##data = dataset.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


### Training

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)

    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'In training accuracy: {acc:.4f}')

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

### Evaluating

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
