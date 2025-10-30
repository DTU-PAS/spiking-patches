"""Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import MLP
from torch_geometric.nn import PointNetConv
from torch_geometric.nn import fps
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import radius

from sp.configs import Config
from sp.configs import TokenizerType
from sp.nn.embeddings import PolarityEmbedding
from sp.nn.embeddings import TokenEmbedding


class PCNClassifier(nn.Module):
    def __init__(self, config: Config, num_classes: int):
        super().__init__()

        dim1 = config.pcn.dim1
        dim2 = config.pcn.dim2
        dim3 = config.pcn.dim3
        fps1 = config.pcn.fps1
        fps2 = config.pcn.fps2
        max_neighbours = config.pcn.max_num_neighbors
        radius1 = config.pcn.radius1
        radius2 = config.pcn.radius2

        if config.tokenizer == TokenizerType.none:
            self.embedding = PolarityEmbedding(dim1)
        else:
            self.embedding = TokenEmbedding(config, dim1)

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(fps1, radius1, max_neighbours, MLP([dim1 + 3, dim1, dim1, dim2]))
        self.sa2_module = SAModule(fps2, radius2, max_neighbours, MLP([dim2 + 3, dim2, dim2, dim3]))
        self.sa3_module = GlobalSAModule(MLP([dim3 + 3, dim3, dim3, dim3]))

        self.classifier = MLP([dim3, dim3, dim3, num_classes], dropout=config.dropout, norm=None)

    def forward(self, data: Batch) -> torch.Tensor:
        x = self.embedding(data.x)
        sa0_out = (x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x = self.sa3_module(*sa2_out)
        return self.classifier(x)


class SAModule(torch.nn.Module):
    def __init__(self, ratio: float, r: float, max_neightbours: int, nn: nn.Module):
        super().__init__()
        self.max_neighbours = max_neightbours
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.max_neighbours)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        return x
