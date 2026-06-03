"""
graph/spd.py — Shortest Path Distance matrix (Floyd-Warshall).
Unreachable pairs -> max_dist + 1.
"""
import torch


def compute_shortest_path_matrix(adj: torch.Tensor, max_dist: int = 5) -> torch.Tensor:
    adj_bin = (adj > 0).float()
    n = adj_bin.shape[0]
    dist = torch.full((n, n), float("inf"), device=adj.device)
    dist.fill_diagonal_(0)
    dist[adj_bin == 1] = 1
    for k in range(n):
        dist = torch.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    dist[dist > max_dist] = max_dist + 1
    return dist.long()
