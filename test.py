from torch_geometric.datasets import Planetoid

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print(data.y.type())
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    if epoch % 100 == 0:
        print(out.size())
        print(data.y.size())
    
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()

#print(data.train_mask)
#print(dataset.num_classes)
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred == data.y).sum()
acc = int(correct) / int((len(data.y)))
print(f'Accuracy: {acc:.4f}')