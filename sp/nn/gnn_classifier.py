import torch
from torch import nn
from torch.nn.functional import elu
from torch_geometric.data import Batch
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import MaxAggregation
from torch_geometric.nn import SplineConv
from torch_geometric.nn import fps
from torch_geometric.nn import radius_graph
from torch_geometric.transforms import Cartesian

from sp.configs import Config
from sp.configs import TokenizerType
from sp.nn.embeddings import PolarityEmbedding
from sp.nn.embeddings import TokenEmbedding


class GNNClassifier(nn.Module):
    def __init__(
        self,
        config: Config,
        num_classes: int,
    ):
        super().__init__()

        bias = config.gnn.bias
        dim1 = config.gnn.dim1
        dim2 = config.gnn.dim2
        dim3 = config.gnn.dim3
        kernel_size = config.gnn.kernel_size
        root_weight = config.gnn.root_weight

        num_pos = 3

        self.cartesian1 = Cartesian(norm=True, cat=False, max_value=config.gnn.node_radius)
        self.cartesian2 = Cartesian(norm=True, cat=False, max_value=config.gnn.pool_radius)

        if config.tokenizer == TokenizerType.none:
            self.embedding = PolarityEmbedding(dim1)
        else:
            self.embedding = TokenEmbedding(config, dim1)

        self.conv1 = SplineConv(dim1, dim2, dim=num_pos, bias=bias, kernel_size=kernel_size, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=dim2)
        self.conv2 = SplineConv(dim2, dim2, dim=num_pos, bias=bias, kernel_size=kernel_size, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=dim2)

        self.conv3 = SplineConv(dim2, dim2, dim=num_pos, bias=bias, kernel_size=kernel_size, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=dim2)
        self.conv4 = SplineConv(dim2, dim2, dim=num_pos, bias=bias, kernel_size=kernel_size, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=dim2)

        self.conv5 = SplineConv(dim2, dim3, dim=num_pos, bias=bias, kernel_size=kernel_size, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=dim3)
        self.pool5 = Pool(config, dim3)

        self.conv6 = SplineConv(dim3, dim3, dim=num_pos, bias=bias, kernel_size=kernel_size, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=dim3)
        self.conv7 = SplineConv(dim3, dim3, dim=num_pos, bias=bias, kernel_size=kernel_size, root_weight=root_weight)
        self.norm7 = BatchNorm(in_channels=dim3)

        self.final_pool = MaxAggregation()
        self.fc = nn.Linear(dim3, out_features=num_classes, bias=bias)

    def forward(self, data: Batch):
        data = self.cartesian1(data)

        x = self.embedding(data.x)

        data.edge_attr = data.edge_attr.to(x.dtype)

        x = elu(self.conv1(x, data.edge_index, data.edge_attr))
        x = self.norm1(x)
        x = elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = self.norm2(x)

        x_sc = x
        x = elu(self.conv3(x, data.edge_index, data.edge_attr))
        x = self.norm3(x)
        x = elu(self.conv4(x, data.edge_index, data.edge_attr))
        x = self.norm4(x)
        x = x + x_sc

        x = elu(self.conv5(x, data.edge_index, data.edge_attr))
        x = self.norm5(x)
        data_pooled = self.pool5(x, pos=data.pos, batch=data.batch)

        data_pooled = self.cartesian2(data_pooled)
        data_pooled.edge_attr = data_pooled.edge_attr.to(x.dtype)

        x_sc = data_pooled.x
        x = data_pooled.x
        x = elu(self.conv6(x, data_pooled.edge_index, data_pooled.edge_attr))
        x = self.norm6(x)
        x = elu(self.conv7(x, data_pooled.edge_index, data_pooled.edge_attr))
        x = self.norm7(x)
        x = x + x_sc

        x = self.final_pool(x, data_pooled.batch)
        return self.fc(x)


class Pool(torch.nn.Module):
    def __init__(self, config: Config, dim: int):
        super().__init__()
        self.fps_ratio = config.gnn.fps_ratio
        self.radius = config.gnn.pool_radius

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> Batch:
        indices = fps(pos, batch=batch, ratio=self.fps_ratio, random_start=self.training)
        sampled_x = x[indices]
        sampled_pos = pos[indices]
        sampled_batch = batch[indices]
        edge_index = radius_graph(sampled_pos, r=self.radius, batch=sampled_batch)
        data = Batch(x=sampled_x, pos=sampled_pos, batch=sampled_batch, edge_index=edge_index)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(fps_ratio={self.fps_ratio}, radius={self.radius})"
