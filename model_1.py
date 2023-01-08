from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
import numpy as np


class GCN_layer(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        #         torch.manual_seed(1234567)
        self.conv1 = GCNConv(num_features, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class DCB(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        #         torch.manual_seed(1234567)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(2 * hidden_channels, hidden_channels)
        self.conv3 = GCNConv(3 * hidden_channels, hidden_channels)
        self.conv4 = GCNConv(4 * hidden_channels, hidden_channels)
        self.conv5 = GCNConv(5 * hidden_channels, hidden_channels)


    def forward(self, x, edge_index):  # x is the feature matrix of the nodes, edge_index is the the adjacency matrix
        x_cat = x

        # first layer
        x = F.relu(x + self.conv1(x_cat, edge_index))  # residual connection before relu, x's dim: num_hidden_features
        x_cat = torch.cat((x_cat, x), 1)  # feature concatenation (horizontally)

        # second layer
        x = F.relu(x + self.conv2(x_cat, edge_index))
        x_cat = torch.cat((x_cat, x), 1)

        # third layer
        x = F.relu(x + self.conv3(x_cat, edge_index))
        x_cat = torch.cat((x_cat, x), 1)

        # fourth layer
        x = F.relu(x + self.conv4(x_cat, edge_index))
        x_cat = torch.cat((x_cat, x), 1)

        # fifth layer
        x = F.relu(x + self.conv5(x_cat, edge_index))
        x_cat = torch.cat((x_cat, x), 1)

        return x_cat  # concatenated features matrix


class MDC_GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        #         torch.manual_seed(7)
        self.gcn1 = GCN_layer(num_features, hidden_channels)
        self.dcb1 = DCB(num_features, hidden_channels)
        self.gcn2 = GCN_layer(6 * hidden_channels, hidden_channels)
        self.dcb2 = DCB(num_features, hidden_channels)
        self.gcn3 = GCN_layer(6 * hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.dcb1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = self.dcb2(x, edge_index)
        x = self.gcn3(x, edge_index)
        return x
