import torch
import torch.nn as nn
import torch.nn.functional as F

from TinyCardioUNet.graph import imu_graph_relationship_all

class conv_block(nn.Module):
    r"""
    https://github.com/LeeJunHyun/Image_Segmentation/blob/5e9da9395c52b119d55dfc6532c34ac0e88f446e/network.py#L29-L44
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    r"""
    https://github.com/LeeJunHyun/Image_Segmentation/blob/5e9da9395c52b119d55dfc6532c34ac0e88f446e/network.py#L46-L58
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.up(x)


class BottleneckGraphSAGE(nn.Module):
    r"""GraphSAGE (mean aggregator) over the C bottleneck channels, per the
    TinyCardioUNet architecture: at the (C, N/8) bottleneck, each of the C
    channels is a graph node whose feature is its length-(N/8) temporal
    vector. For node v with neighbors N(v):

        h_{N(v)} = mean_{u in N(v)} h_u                  (neighbor aggregation)
        h_v'     = W1 h_v + W2 h_{N(v)} + b               (self + neighbor transform)
        h_v'     = h_v' / ||h_v'||_2                      (L2-normalize)

    with no activation in between. Implemented as a matmul with a fixed,
    precomputed row-normalized adjacency matrix instead of torch_geometric's
    scatter-based SAGEConv: the graph is tiny and static, so this is both
    fully deterministic (no scatter-order nondeterminism) and uses plain
    nn.Linear for W1/W2, so VBMF/SVD compression (decompose_tinycardiounet_vbmf.py)
    applies to them directly -- PyG's SAGEConv uses its own Linear class that
    is NOT an nn.Linear subclass, which used to make that compression branch
    silently no-op.
    """

    def __init__(self, num_channels, feature_dim):
        super().__init__()
        G = imu_graph_relationship_all()
        self.lin_self = nn.Linear(feature_dim, feature_dim)              # W1 (+ shared bias b)
        self.lin_neighbor = nn.Linear(feature_dim, feature_dim, bias=False)  # W2
        self.register_buffer("adj", self._build_adjacency(G, num_channels))

    @staticmethod
    def _build_adjacency(G, num_nodes):
        """Row-normalized adjacency: adj[i] @ h == mean of node i's neighbors."""
        adj = torch.zeros(num_nodes, num_nodes)
        for u, v in G.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        return adj / degree

    def forward(self, x):
        # x: (B, C, L) bottleneck feature map, C channels = graph nodes
        neighbor = torch.einsum("ij,bjl->bil", self.adj, x)   # mean of neighbors, per channel
        out = self.lin_self(x) + self.lin_neighbor(neighbor)
        return F.normalize(out, p=2, dim=-1)


class TinyCardioUNet(nn.Module):
    def __init__(self, in_channels=3, seq_len=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enc1 = conv_block(in_channels, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.enc4 = conv_block(64, 128)

        self.graph = BottleneckGraphSAGE(num_channels=128, feature_dim=seq_len // 8)

        self.dec4_up = up_conv(128, 64)
        self.dec4_conv = conv_block(128, 64)
        self.dec3_up = up_conv(64, 32)
        self.dec3_conv = conv_block(64, 32)
        self.dec2_up = up_conv(32, 16)
        self.dec2_conv = conv_block(32, 16)
        self.dec1 = nn.Conv1d(16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool1d(e1, 2, 2))
        e3 = self.enc3(F.max_pool1d(e2, 2, 2))
        e4 = self.enc4(F.max_pool1d(e3, 2, 2))
        e_graph = self.graph(e4)

        d4_up = self.dec4_up(e_graph)
        d4 = self.dec4_conv(torch.cat([d4_up, e3], dim=1))

        d3_up = self.dec3_up(d4)
        d3 = self.dec3_conv(torch.cat([d3_up, e2], dim=1))

        d2_up = self.dec2_up(d3)
        d2 = self.dec2_conv(torch.cat([d2_up, e1], dim=1))

        d1 = self.dec1(d2)
        return d1
