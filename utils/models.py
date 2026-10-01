"""
Model definitions for TASK3.

Includes:
- OpticalResponseGINE
- OpticalResponseGCN
- OpticalResponseGraphSAGE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, GCNConv, SAGEConv, global_mean_pool

# ============================================================
# OpticalResponseGINE model
# ============================================================

class OpticalResponseGINE(nn.Module):
    """
    GINE model for optical-response spectrum prediction.

    Supports:
    - baseline graph: node dim 11, edge dim 16
    - enhanced graph: node dim 19, edge dim 24
    - optional graph_attr
    """

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        out_dim,
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.10,
        graph_attr_dim=0,
        use_graph_attr=False,
    ):
        super().__init__()

        self.node_in_dim = int(node_in_dim)
        self.edge_in_dim = int(edge_in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.graph_attr_dim = int(graph_attr_dim)
        self.use_graph_attr = bool(use_graph_attr and self.graph_attr_dim > 0)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

            conv = GINEConv(
                nn=mlp,
                edge_dim=self.edge_in_dim,
                train_eps=True,
            )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(self.hidden_dim))

        head_in_dim = self.hidden_dim

        if self.use_graph_attr:
            head_in_dim += self.graph_attr_dim

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.out_dim),
        )

    def forward(
        self,
        data=None,
        x=None,
        edge_index=None,
        edge_attr=None,
        batch=None,
        graph_attr=None,
        **kwargs,
    ):
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch = data.batch
            graph_attr = getattr(data, "graph_attr", None)

        if x is None or edge_index is None or edge_attr is None:
            raise RuntimeError("OpticalResponseGINE requires x, edge_index, and edge_attr.")

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res

        g = global_mean_pool(h, batch)

        if self.use_graph_attr and graph_attr is not None:
            graph_attr = graph_attr.view(g.size(0), -1).to(g.device)
            g = torch.cat([g, graph_attr], dim=1)

        out = self.head(g)
        return out


def build_optical_response_gine(
    node_in_dim,
    edge_in_dim,
    out_dim,
    hidden_dim=192,
    latent_dim=256,
    num_layers=4,
    dropout=0.10,
    graph_attr_dim=0,
    use_graph_attr=False,
):
    return OpticalResponseGINE(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        dropout=dropout,
        graph_attr_dim=graph_attr_dim,
        use_graph_attr=use_graph_attr,
    )


# ------------------------------------------------------------
# Baseline models extracted from TASK3 notebook
# ------------------------------------------------------------
class OpticalResponseGCN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim=None,
        out_dim=2001,
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.10,
        **kwargs,
    ):
        super().__init__()

        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res

        g = global_mean_pool(h, batch)
        out = self.graph_head(g)

        return out

class OpticalResponseGraphSAGE(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim=None,
        out_dim=2001,
        hidden_dim=192,
        latent_dim=256,
        num_layers=4,
        dropout=0.10,
        **kwargs,
    ):
        super().__init__()

        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self.node_encoder(x)

        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_res

        g = global_mean_pool(h, batch)
        out = self.graph_head(g)

        return out
