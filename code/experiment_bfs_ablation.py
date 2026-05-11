#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C2 - BFS-only Ablation within Quantum Walk Subgraph
Reviewer Opus 4.7: "The MRR jump 0.919 -> 1.000 may be mostly due to restricting
search space to local subgraph rather than quantum interference."

Within the same 50-node BFS subgraph, compare:
  (a) BFS distance ranking
  (b) Node degree ranking
  (c) Local PageRank ranking
  (d) CTQW classical simulation (our method)

If CTQW doesn't win cleanly, reframe the contribution.
"""

import sys
import time
import random
import statistics
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from topology_aware_sdm import load_graph

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def bfs_subgraph(query_id, neighbors_map, max_size=50):
    """BFS extraction — same as quantum_walk.py."""
    visited = {query_id}
    queue = [query_id]
    order = [query_id]  # keep order for BFS distance computation
    bfs_dist = {query_id: 0}
    while queue and len(visited) < max_size:
        current = queue.pop(0)
        for nbr in neighbors_map.get(current, []):
            if nbr not in visited and len(visited) < max_size:
                visited.add(nbr)
                queue.append(nbr)
                order.append(nbr)
                bfs_dist[nbr] = bfs_dist[current] + 1
    return order, bfs_dist


def rank_by_bfs_distance(query_id, subgraph_ids, bfs_dist):
    """Rank nodes by BFS distance from query (closest first)."""
    # Tie-break with node ID for stability
    ranked = sorted(
        [(bfs_dist.get(nid, float('inf')), nid) for nid in subgraph_ids if nid != query_id]
    )
    return [nid for _, nid in ranked]


def rank_by_local_degree(query_id, subgraph_ids, edges):
    """Rank nodes by their degree within the subgraph (highest first)."""
    subgraph_set = set(subgraph_ids)
    degree = defaultdict(int)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src in subgraph_set and tgt in subgraph_set:
            degree[src] += 1
            degree[tgt] += 1
    # Sort descending by degree, break ties by node ID
    ranked = sorted(
        [(-degree.get(nid, 0), nid) for nid in subgraph_ids if nid != query_id]
    )
    return [nid for _, nid in ranked]


def rank_by_local_pagerank(query_id, subgraph_ids, edges, alpha=0.85, n_iter=50):
    """Rank nodes by personalized PageRank from query within subgraph."""
    N = len(subgraph_ids)
    idx = {nid: i for i, nid in enumerate(subgraph_ids)}
    subgraph_set = set(subgraph_ids)

    # Build adjacency (undirected)
    A = np.zeros((N, N), dtype=np.float64)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src in subgraph_set and tgt in subgraph_set:
            A[idx[src], idx[tgt]] = 1
            A[idx[tgt], idx[src]] = 1

    # Normalize to transition matrix (avoid div by zero)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    P = A / row_sums

    # Personalized PageRank from query
    q_idx = idx[query_id]
    teleport = np.zeros(N)
    teleport[q_idx] = 1.0

    scores = teleport.copy()
    for _ in range(n_iter):
        scores = alpha * (P.T @ scores) + (1 - alpha) * teleport

    # Rank descending (exclude query)
    ranked_with_scores = sorted(
        [(-scores[i], nid) for i, nid in enumerate(subgraph_ids) if nid != query_id]
    )
    return [nid for _, nid in ranked_with_scores]


def rank_by_ctqw(query_id, subgraph_ids, edges, t=0.5):
    """Rank nodes by CTQW amplitude (our method)."""
    from scipy.linalg import expm

    N = len(subgraph_ids)
    idx = {nid: i for i, nid in enumerate(subgraph_ids)}
    subgraph_set = set(subgraph_ids)

    A = np.zeros((N, N), dtype=np.complex128)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src in subgraph_set and tgt in subgraph_set:
            A[idx[src], idx[tgt]] = 1
            A[idx[tgt], idx[src]] = 1

    H = -A
    psi_0 = np.zeros(N, dtype=np.complex128)
    psi_0[idx[query_id]] = 1.0
    U = expm(-1j * H * t)
    psi_t = U @ psi_0
    probs = np.abs(psi_t) ** 2

    # Rank descending by probability (exclude query)
    ranked = sorted(
        [(-float(probs[i]), nid) for i, nid in enumerate(subgraph_ids) if nid != query_id]
    )
    return [nid for _, nid in ranked]


def measure_mrr_from_ranking(ranked_ids, true_neighbors):
    """Compute MRR and Recall@5 from a ranked list."""
    rr = 0.0
    for rank, nid in enumerate(ranked_ids, 1):
        if nid in true_neighbors:
            rr = 1.0 / rank
            break

    top5 = set(ranked_ids[:5])
    r5 = len(top5 & true_neighbors) / len(true_neighbors) if true_neighbors else 0.0
    return rr, r5


def main():
    print("=" * 65)
    print("C2 - BFS-only Ablation within Quantum Walk Subgraph")
    print("Testing: is CTQW contribution > simple baselines on same subgraph?")
    print("=" * 65)

    repo_root = Path(__file__).parent.parent
    nodes, edges = load_graph(
        str(repo_root / 'data' / 'graph.jsonl'),
        str(repo_root / 'data' / 'edges.jsonl'),
    )
    print(f"\nGraph: {len(nodes)} nodes, {len(edges)} edges")

    # Build neighbors
    neighbors_map = defaultdict(list)
    true_neighbors = defaultdict(set)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src and tgt:
            neighbors_map[src].append(tgt)
            neighbors_map[tgt].append(src)
            true_neighbors[src].add(tgt)
            true_neighbors[tgt].add(src)

    # Select query nodes (same protocol as other experiments)
    query_candidates = [n for n in nodes if len(true_neighbors[n['id']]) >= 2]
    random.seed(0)
    random.shuffle(query_candidates)
    query_nodes = query_candidates[:20]  # fewer queries for speed

    methods = {
        'bfs_distance': [],
        'local_degree': [],
        'local_pagerank': [],
        'ctqw_t0.5': [],
    }
    methods_r5 = {k: [] for k in methods}

    print(f"\nRunning {len(query_nodes)} queries with subgraph_size=50...\n")
    print(f"{'Query':<20} {'BFS':<8} {'Degree':<8} {'PageRank':<10} {'CTQW':<8}")
    print("-" * 65)

    for qnode in query_nodes:
        qid = qnode['id']
        true_nbrs = true_neighbors[qid]

        subgraph, bfs_dist = bfs_subgraph(qid, neighbors_map, max_size=50)
        if len(subgraph) < 2:
            continue

        # Method a: BFS distance
        rank_bfs = rank_by_bfs_distance(qid, subgraph, bfs_dist)
        mrr_bfs, r5_bfs = measure_mrr_from_ranking(rank_bfs, true_nbrs)

        # Method b: local degree
        rank_deg = rank_by_local_degree(qid, subgraph, edges)
        mrr_deg, r5_deg = measure_mrr_from_ranking(rank_deg, true_nbrs)

        # Method c: local PageRank
        rank_pr = rank_by_local_pagerank(qid, subgraph, edges)
        mrr_pr, r5_pr = measure_mrr_from_ranking(rank_pr, true_nbrs)

        # Method d: CTQW (our method)
        rank_qw = rank_by_ctqw(qid, subgraph, edges, t=0.5)
        mrr_qw, r5_qw = measure_mrr_from_ranking(rank_qw, true_nbrs)

        methods['bfs_distance'].append(mrr_bfs)
        methods['local_degree'].append(mrr_deg)
        methods['local_pagerank'].append(mrr_pr)
        methods['ctqw_t0.5'].append(mrr_qw)

        methods_r5['bfs_distance'].append(r5_bfs)
        methods_r5['local_degree'].append(r5_deg)
        methods_r5['local_pagerank'].append(r5_pr)
        methods_r5['ctqw_t0.5'].append(r5_qw)

        print(f"{qid[:18]:<20} {mrr_bfs:<8.3f} {mrr_deg:<8.3f} {mrr_pr:<10.3f} {mrr_qw:<8.3f}")

    print("\n" + "=" * 65)
    print("RESULTS — Within-subgraph MRR (N=50)")
    print("=" * 65)

    print(f"\n{'Method':<25} {'MRR mean':<12} {'MRR std':<12} {'Recall@5':<12}")
    print("-" * 65)
    for name in ['bfs_distance', 'local_degree', 'local_pagerank', 'ctqw_t0.5']:
        mrrs = methods[name]
        r5s = methods_r5[name]
        if not mrrs:
            continue
        mrr_mean = statistics.mean(mrrs)
        mrr_std = statistics.stdev(mrrs) if len(mrrs) > 1 else 0.0
        r5_mean = statistics.mean(r5s)
        print(f"{name:<25} {mrr_mean:<12.4f} {mrr_std:<12.4f} {r5_mean:<12.4f}")

    # Head-to-head: is CTQW significantly better than best baseline?
    best_baseline_name = max(
        ['bfs_distance', 'local_degree', 'local_pagerank'],
        key=lambda k: statistics.mean(methods[k]) if methods[k] else 0
    )
    print(f"\nBest non-CTQW baseline: {best_baseline_name}")

    if methods['ctqw_t0.5'] and methods[best_baseline_name]:
        diffs = [
            qw - bl for qw, bl in zip(methods['ctqw_t0.5'], methods[best_baseline_name])
        ]
        if diffs and statistics.stdev(diffs) > 0:
            t_stat = statistics.mean(diffs) / (statistics.stdev(diffs) / (len(diffs) ** 0.5))
            print(f"Paired t-test CTQW vs {best_baseline_name}:")
            print(f"  Mean paired difference: {statistics.mean(diffs):.4f}")
            print(f"  t-statistic: {t_stat:.2f} (df={len(diffs)-1})")

            # Critical values for df=19 (or nearby)
            if abs(t_stat) > 3.883:
                sig = "p < 0.001 (highly significant)"
            elif abs(t_stat) > 2.861:
                sig = "p < 0.01  (significant)"
            elif abs(t_stat) > 2.093:
                sig = "p < 0.05  (significant)"
            else:
                sig = "p > 0.05  (NOT significant — CTQW contribution is marginal)"
            print(f"  {sig}")

    # Interpretation
    print("\n" + "=" * 65)
    print("INTERPRETATION")
    print("=" * 65)
    mrr_bfs = statistics.mean(methods['bfs_distance']) if methods['bfs_distance'] else 0
    mrr_qw = statistics.mean(methods['ctqw_t0.5']) if methods['ctqw_t0.5'] else 0
    if mrr_bfs >= 0.95 * mrr_qw:
        print("CTQW contribution is MARGINAL.")
        print("Most of the benefit comes from restricting search to the local subgraph.")
        print("Paper should reframe: 'local subgraph extraction + simple ranking'")
        print("already achieves most of the gain.")
    elif mrr_qw > mrr_bfs + 0.05:
        print("CTQW adds substantial value over simple baselines (+{:.3f} MRR).".format(mrr_qw - mrr_bfs))
        print("The quantum interference pattern IS contributing to the ranking.")
    else:
        print("CTQW and BFS-distance are close; CTQW advantage is small but positive.")

    # Write CSV
    out_csv = repo_root / 'data' / 'bfs-ablation-m1.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("query_idx,bfs_distance_mrr,local_degree_mrr,local_pagerank_mrr,ctqw_mrr\n")
        for i in range(len(methods['ctqw_t0.5'])):
            f.write(f"{i},{methods['bfs_distance'][i]:.4f},{methods['local_degree'][i]:.4f},"
                    f"{methods['local_pagerank'][i]:.4f},{methods['ctqw_t0.5'][i]:.4f}\n")
    try:
        rel = out_csv.relative_to(Path.cwd())
        print(f"\n[CSV] Written to {rel}")
    except ValueError:
        print(f"\n[CSV] Written to {out_csv}")


if __name__ == '__main__':
    main()
