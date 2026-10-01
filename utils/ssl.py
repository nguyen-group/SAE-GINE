"""
Self-supervised pretraining utilities.

Notebook SSL pretext task:
corrupted graph -> graph descriptor prediction
(target = node mean/std + edge mean/std)
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

def graph_desc_target(batch):
    B = int(batch.num_graphs)

    node_mean = global_mean_pool(batch.x, batch.batch, size=B)
    node_sq   = global_mean_pool(batch.x * batch.x, batch.batch, size=B)
    node_std  = torch.sqrt(torch.clamp(node_sq - node_mean * node_mean, min=1e-8))

    edge_batch = batch.batch[batch.edge_index[0]]
    edge_mean = global_mean_pool(batch.edge_attr, edge_batch, size=B)
    edge_sq   = global_mean_pool(batch.edge_attr * batch.edge_attr, edge_batch, size=B)
    edge_std  = torch.sqrt(torch.clamp(edge_sq - edge_mean * edge_mean, min=1e-8))

    return torch.cat([node_mean, node_std, edge_mean, edge_std], dim=1)

def corrupt_graph(batch, p_node=0.15, noise=0.02):
    x = batch.x.clone()
    e = batch.edge_attr.clone()

    mask = torch.rand(x.size(0), device=x.device) < p_node
    x[mask] = 0.0

    x = x + noise * torch.randn_like(x)
    e = e + noise * torch.randn_like(e)

    return x, e

def infer_ssl_out_dim_from_graph(graph):
    """Return 2 * node_dim + 2 * edge_dim for graph descriptor prediction."""
    node_dim = int(graph.x.shape[1])
    edge_dim = int(graph.edge_attr.shape[1])
    return 2 * node_dim + 2 * edge_dim


def check_ssl_graph_dims(graph, expected_node_dim=19, expected_edge_dim=24):
    """Small sanity check used before SSL pretraining."""
    node_dim = int(graph.x.shape[1])
    edge_dim = int(graph.edge_attr.shape[1])
    if node_dim != int(expected_node_dim) or edge_dim != int(expected_edge_dim):
        raise RuntimeError(
            f"Expected graph {expected_node_dim}/{expected_edge_dim}, "
            f"got {node_dim}/{edge_dim}."
        )
    return {"node_dim": node_dim, "edge_dim": edge_dim, "ssl_out_dim": 2 * node_dim + 2 * edge_dim}
