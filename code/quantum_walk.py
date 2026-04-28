#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Continuous-Time Quantum Walk (CTQW) refinement for graph retrieval.

Companion to topology_aware_sdm.py.
Paper DOI: 10.5281/zenodo.19645323
License: MIT
"""

import numpy as np
from collections import defaultdict
from scipy.linalg import expm


def extract_subgraph(
    query_id: str,
    neighbors_map: dict[str, list[str]],
    max_size: int = 50,
) -> list[str]:
    """
    BFS extraction of a bounded subgraph around the query node.

    Args:
        query_id: starting node ID
        neighbors_map: dict node_id -> list of neighbor node_ids
        max_size: cap on subgraph size (default 50, optimal per paper)

    Returns:
        list of node IDs in the subgraph (query first, then BFS order)
    """
    visited = {query_id}
    queue = [query_id]
    order = [query_id]
    while queue and len(visited) < max_size:
        current = queue.pop(0)
        for neighbor in neighbors_map.get(current, []):
            if neighbor not in visited and len(visited) < max_size:
                visited.add(neighbor)
                queue.append(neighbor)
                order.append(neighbor)
    return order


def build_adjacency(
    node_ids: list[str],
    edges: list[dict],
) -> np.ndarray:
    """
    Build symmetric adjacency matrix for the subgraph.

    Treats graph as undirected for walk purposes.
    Returns N×N complex matrix for direct use with scipy.linalg.expm.
    """
    N = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    A = np.zeros((N, N), dtype=np.complex128)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src in idx and tgt in idx:
            i, j = idx[src], idx[tgt]
            A[i, j] = 1
            A[j, i] = 1
    return A


def quantum_walk_query(
    query_id: str,
    subgraph_ids: list[str],
    A: np.ndarray,
    t: float = 0.5,
) -> np.ndarray:
    """
    Run CTQW starting from query_id on the given subgraph.

    |ψ(t)⟩ = exp(-iHt) |ψ(0)⟩,  H = -A.

    Args:
        query_id: starting node (must be in subgraph_ids)
        subgraph_ids: list of node IDs defining the subgraph
        A: N×N adjacency matrix (from build_adjacency)
        t: evolution time (default 0.5, optimal per paper for M ≤ 100)

    Returns:
        1D numpy array of length N with probability |⟨v_j|ψ(t)⟩|² for each node
    """
    N = len(subgraph_ids)
    idx = {nid: i for i, nid in enumerate(subgraph_ids)}
    q_idx = idx[query_id]

    psi_0 = np.zeros(N, dtype=np.complex128)
    psi_0[q_idx] = 1.0

    H = -A
    U = expm(-1j * H * t)
    psi_t = U @ psi_0
    probs = np.abs(psi_t) ** 2
    return probs


def quantum_walk_top_k(
    query_id: str,
    neighbors_map: dict[str, list[str]],
    edges: list[dict],
    k: int = 10,
    subgraph_size: int = 50,
    t: float = 0.5,
) -> list[tuple[float, str]]:
    """
    Top-level CTQW retrieval: extract subgraph, build adjacency, evolve, rank.

    Returns:
        list of (probability, node_id) sorted descending by probability,
        excluding the query node itself.
    """
    subgraph = extract_subgraph(query_id, neighbors_map, subgraph_size)
    if len(subgraph) < 2:
        return []
    A = build_adjacency(subgraph, edges)
    probs = quantum_walk_query(query_id, subgraph, A, t)

    results = [
        (float(probs[i]), subgraph[i])
        for i in range(len(subgraph))
        if subgraph[i] != query_id
    ]
    results.sort(key=lambda x: -x[0])
    return results[:k]


def hybrid_retrieval(
    query_id: str,
    neighbors_map: dict[str, list[str]],
    edges: list[dict],
    ta_sdm_addresses: dict[str, int],
    k: int = 10,
    refinement_threshold_k: int = 50,
) -> list[tuple[str, str]]:
    """
    Hybrid retrieval: TA-SDM for first-stage, CTQW for refinement.

    Use TA-SDM's fast Hamming ranking for the full graph, then refine the
    top candidates via quantum walk on their local subgraph.

    Returns:
        list of (source, node_id) where source is 'ctqw' or 'ta-sdm'.
    """
    # Defer importing TA-SDM to keep this module optional
    from topology_aware_sdm import top_k

    # Stage 1: TA-SDM candidates
    if query_id not in ta_sdm_addresses:
        return []

    ta_results = top_k(
        ta_sdm_addresses[query_id],
        ta_sdm_addresses,
        k=refinement_threshold_k,
        exclude={query_id},
    )

    # Stage 2: CTQW refinement on BFS subgraph
    ctqw_results = quantum_walk_top_k(
        query_id, neighbors_map, edges, k=k,
        subgraph_size=refinement_threshold_k,
    )

    # Combine: CTQW results have priority, falling back to TA-SDM
    ctqw_ids = {nid for _, nid in ctqw_results}
    combined = [('ctqw', nid) for _, nid in ctqw_results]
    for _, nid in ta_results:
        if nid not in ctqw_ids and len(combined) < k:
            combined.append(('ta-sdm', nid))
    return combined[:k]


if __name__ == '__main__':
    import argparse
    import json
    import sys
    from pathlib import Path

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='Classical quantum walk retrieval')
    parser.add_argument('--nodes', default='data/graph.jsonl', help='Nodes JSONL path')
    parser.add_argument('--edges', default='data/edges.jsonl', help='Edges JSONL path')
    parser.add_argument('--query-id', required=True, help='Query node ID')
    parser.add_argument('--subgraph-size', type=int, default=50,
                        help='BFS subgraph size (default: 50, optimal per paper)')
    parser.add_argument('--t', type=float, default=0.5,
                        help='Evolution time (default: 0.5)')
    parser.add_argument('--k', type=int, default=10, help='Top-K results')
    args = parser.parse_args()

    # Load graph
    nodes, edges = [], []
    with open(args.nodes, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    nodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    with open(args.edges, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    edges.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    neighbors = defaultdict(list)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src and tgt:
            neighbors[src].append(tgt)
            neighbors[tgt].append(src)

    print(f"[CTQW] Loaded {len(nodes)} nodes, {len(edges)} edges")
    print(f"[CTQW] Query: {args.query_id}, subgraph size: {args.subgraph_size}, t: {args.t}")

    results = quantum_walk_top_k(
        args.query_id, neighbors, edges,
        k=args.k, subgraph_size=args.subgraph_size, t=args.t,
    )

    nodes_by_id = {n['id']: n for n in nodes}
    print(f"\n[CTQW] Top-{args.k} results:")
    for i, (prob, nid) in enumerate(results, 1):
        node = nodes_by_id.get(nid, {})
        title = node.get('title') or node.get('label') or ''
        print(f"  {i:2d}. {nid:20s} (prob={prob:.4f})  {title[:60]}")
