"""Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py"""

import math

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import MLP
from torch_geometric.nn import PointNetConv
from torch_geometric.nn import fps
from torch_geometric.nn import max_pool_x
from torch_geometric.nn import radius
from torch_geometric.nn import voxel_grid

from sp.configs import Config
from sp.configs import TokenizerType
from sp.loaders import load_dimensions
from sp.nn.embeddings import PolarityEmbedding
from sp.nn.embeddings import TokenEmbedding


class PCNDetector(nn.Module):
    def __init__(self, config: Config):
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
        self.sa3_module = GridPool(config, MLP([dim3 + 3, dim3, dim3, dim3]))

    def forward(self, data: Batch, batch_size: int) -> torch.Tensor:
        x = self.embedding(data.x)
        sa0_out = (x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        x = self.sa3_module(*sa2_out, batch_size)
        return x


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


class GridPool(torch.nn.Module):
    def __init__(self, config: Config, nn):
        super().__init__()
        self.nn = nn

        self.hidden_size = config.pcn.dim3

        height, width = load_dimensions(config.dataset)
        grid_size = config.pcn.grid_size
        patch_size = config.patch_size
        self.grid_size = (grid_size, grid_size)
        if config.tokenizer == TokenizerType.none:
            self.num_rows = math.ceil(height / grid_size)
            self.num_cols = math.ceil(width / grid_size)
            self.end = [width - 1, height - 1]
        else:
            num_input_rows = math.ceil(height / patch_size)
            num_input_cols = math.ceil(width / patch_size)
            self.num_rows = math.ceil(num_input_rows / grid_size)
            self.num_cols = math.ceil(num_input_cols / grid_size)
            self.end = [num_input_cols - 1, num_input_rows - 1]
        self.num_cells = self.num_rows * self.num_cols

    def forward(self, x, pos, batch, batch_size: int) -> torch.Tensor:
        x = self.nn(torch.cat([x, pos], dim=1))

        cluster = voxel_grid(pos[:, :2], batch=batch, size=self.grid_size, start=0, end=self.end)
        x, _ = max_pool_x(cluster, x, batch, batch_size=batch_size, size=self.num_cells)
        x = x.reshape(batch_size, self.num_rows, self.num_cols, self.hidden_size)

        return x
