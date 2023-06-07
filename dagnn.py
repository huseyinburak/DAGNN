#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:34:17 2023

@author: hbonen
"""

import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.manifold import TSNE
from torch_geometric.nn import DataParallel
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, to_undirected
from torch_scatter import scatter_add
import argparse
import matplotlib.pyplot as plt


# Set random seed for reproducibility
torch.manual_seed(42)

# Load the Cora dataset
dataset = Planetoid(root='./cora', name='Cora')

data = dataset[0]
x, y = data.x, data.y

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = torch.tensor(y_encoded)

# Split the data into train, validation, and test sets
train_idx, test_idx, train_y, test_y = train_test_split(range(len(y_encoded)), y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)

# Create the adjacency matrix (A)
edge_index = data.edge_index
adj = torch.zeros((x.shape[0], x.shape[0]))
adj[edge_index[0], edge_index[1]] = 1

# Convert data to PyTorch tensors
x = torch.tensor(x, dtype=torch.float32)
adj = adj.to(torch.float32)

parser = argparse.ArgumentParser()
parser.add_argument('--K', type=int, default=10)
args = parser.parse_args()

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)


        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()
    
    
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_layers, num_classes, dropout_rate):
        super(GCN, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(num_features, hidden_layers[0]))
        self.dropout_rate = dropout_rate
        self.bn = []
        self.bn.append(torch.nn.BatchNorm1d(hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.linears.append(torch.nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.bn.append(torch.nn.BatchNorm1d(hidden_layers[i]))
        self.linears.append(torch.nn.Linear(hidden_layers[-1], num_classes))
        # self.bn = torch.nn.BatchNorm1d(hidden_layers[-1])
        self.prop = Prop(num_classes, args.K)
        # self.dropout = dropout
        
    def reset_parameters(self):
        for l in self.linears:
            self.linears[l].reset_parameters()
            self.bn[l].reset_parameters()
        self.prop.reset_parameters()

    
    def forward(self, x, edge_index):
        for l in range(len(self.linears[:-1])):
            x = self.linears[l](x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = F.relu(self.bn[l](x))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linears[-1](x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)

hidden_node = 16
hidden_layer = 5
accs = []

for i in range(1, hidden_layer +1 ):
    hidden_layers = []
    for j in range(1,i+1):
        hidden_layers.append(hidden_layer)

    dropout_rate = 0.
    model = GCN(num_features=x.shape[1], hidden_layers=hidden_layers, num_classes=dataset.num_classes, dropout_rate=dropout_rate)
    
    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Training loop
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)  # Pass the entire feature tensor
        loss = F.nll_loss(out[train_idx], train_y)  # Only compute loss for the training nodes
        loss.backward()
        optimizer.step()
    
    # Evaluation loop
    def evaluate():
        model.eval()
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[test_idx] == test_y
        test_acc = int(test_correct.sum()) / len(test_idx)
        return test_acc
    
    # Training and evaluation
    best_test_acc = 0
    for epoch in range(1, 201):
        train()
        test_acc = evaluate()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pt')
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')
    
    # Load the best model and evaluate on the test set
    model.load_state_dict(torch.load('best_model.pt'))
    test_acc = evaluate()
    accs.append(test_acc)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Classification report
    model.eval()
    out = model(x, edge_index)
    pred = out.argmax(dim=1)
    pred_labels = label_encoder.inverse_transform(pred.detach().numpy())
    true_labels = label_encoder.inverse_transform(y_encoded.numpy())
    print(classification_report(true_labels, pred_labels))
    
    # Convert the output features to numpy array
    out_features = out.detach().numpy()
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2)
    out_tsne = tsne.fit_transform(out_features)
    
    # Visualize the input and output features using scatter plots
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(out_tsne[:, 0], out_tsne[:, 1], c=y, cmap='tab10')
    plt.title('Output Features')
    
    # Convert the input features to numpy array
    x_features = x.detach().numpy()
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_features)
    
    plt.subplot(122)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap='tab10')
    plt.title('Input Features')
    
    plt.tight_layout()
    plt.show()

layers = range(1, hidden_layer + 1)

plt.figure()
plt.plot(layers, accs, marker='o')
plt.xlabel('Layer')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs. Layer')
plt.show()
