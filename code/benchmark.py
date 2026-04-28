#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: reproduces the principal results of the paper.

Usage:  python code/benchmark.py

Paper DOI: 10.5281/zenodo.19645323
License: MIT
"""

import json
import random
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

# Add sibling modules to path
sys.path.insert(0, str(Path(__file__).parent))

from topology_aware_sdm import (
    simhash, hamming, bind, unbind, node_to_text,
    content_address, topology_address, build_addresses, top_k,
    load_graph,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


ROOT = Path(__file__).parent.parent  # Repo root
DATA = ROOT / 'data'


# ─── MRR / Recall@5 measurement ──────────────────────────────────────────

def build_true_neighbors(edges: list[dict]) -> dict[str, set[str]]:
    """Build the ground-truth neighbor sets from edges."""
    true_neighbors = defaultdict(set)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src and tgt:
            true_neighbors[src].add(tgt)
            true_neighbors[tgt].add(src)
    return true_neighbors


def measure_mrr(
    nodes: list[dict],
    edges: list[dict],
    addresses_fn,
    n_queries: int = 50,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Measure MRR and Recall@5 for graph neighbor retrieval.

    addresses_fn: callable (node) -> binary address int.
    """
    true_neighbors = build_true_neighbors(edges)

    random.seed(seed)
    query_candidates = [n for n in nodes if len(true_neighbors[n['id']]) >= 2]
    random.shuffle(query_candidates)
    query_nodes = query_candidates[:n_queries]

    # Compute addresses for all nodes
    addresses = {n['id']: addresses_fn(n) for n in nodes}

    rrs = []
    recall5_scores = []
    for qnode in query_nodes:
        qid = qnode['id']
        qaddr = addresses[qid]
        true_nbrs = true_neighbors[qid]

        ranked = sorted(
            [(hamming(qaddr, addresses[nid]), nid) for nid in addresses if nid != qid]
        )

        # MRR: reciprocal rank of first true neighbor
        rr = 0.0
        for rank, (_, nid) in enumerate(ranked, 1):
            if nid in true_nbrs:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        # Recall@5
        top5_ids = {nid for _, nid in ranked[:5]}
        r5 = len(top5_ids & true_nbrs) / len(true_nbrs) if true_nbrs else 0.0
        recall5_scores.append(r5)

    return (
        sum(rrs) / len(rrs) if rrs else 0.0,
        sum(recall5_scores) / len(recall5_scores) if recall5_scores else 0.0,
    )


# ─── Main benchmark ──────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 65)
    print("Topology-Aware SDM + Classical Quantum Walk — Benchmark")
    print("DOI: 10.5281/zenodo.19645323")
    print("=" * 65)

    # Load graph
    nodes_path = DATA / 'graph.jsonl'
    edges_path = DATA / 'edges.jsonl'
    if not nodes_path.exists() or not edges_path.exists():
        print(f"\nERROR: graph files not found at {DATA}/")
        print("Expected: graph.jsonl and edges.jsonl")
        sys.exit(1)

    nodes, edges = load_graph(nodes_path, edges_path)
    print(f"\n[LOAD] {len(nodes)} nodes, {len(edges)} edges")

    neighbors_map = defaultdict(list)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src and tgt:
            neighbors_map[src].append(tgt)
            neighbors_map[tgt].append(src)

    # ── Experiment 1: Content-only baseline ──────────────────────────
    print("\n[1] Content-only SimHash baseline (d=256)")
    t0 = time.perf_counter()
    content_addrs = {n['id']: content_address(n, 256) for n in nodes}
    t_content = time.perf_counter() - t0
    print(f"    Built in {t_content*1000:.0f}ms")

    mrr_content, r5_content = measure_mrr(
        nodes, edges, lambda n: content_addrs[n['id']],
    )
    print(f"    MRR = {mrr_content:.3f}   Recall@5 = {r5_content:.3f}")

    # ── Experiment 2: TA-SDM (our contribution) ──────────────────────
    print("\n[2] Topology-Aware SDM (d=256, 1-hop)")
    t0 = time.perf_counter()
    ta_addrs = build_addresses(nodes, edges, bits=256)
    t_ta = time.perf_counter() - t0
    print(f"    Built in {t_ta*1000:.0f}ms")

    mrr_ta, r5_ta = measure_mrr(
        nodes, edges, lambda n: ta_addrs[n['id']],
    )
    print(f"    MRR = {mrr_ta:.3f}   Recall@5 = {r5_ta:.3f}")

    if mrr_content > 0:
        print(f"    Improvement: {mrr_ta / mrr_content:.2f}x MRR, "
              f"{r5_ta / r5_content:.2f}x Recall@5")

    # ── Experiment 3: Dimensionality ablation ────────────────────────
    print("\n[3] Dimensionality ablation (1-hop)")
    print(f"    {'bits':>6}  {'MRR':>8}  {'Recall@5':>8}")
    for bits in [128, 256, 512, 1024]:
        addrs = build_addresses(nodes, edges, bits=bits)
        mrr, r5 = measure_mrr(nodes, edges, lambda n, a=addrs: a[n['id']])
        marker = " ←" if bits == 256 else ""
        print(f"    {bits:>6}  {mrr:>8.3f}  {r5:>8.3f}{marker}")

    # ── Experiment 4: Classical quantum walk refinement ──────────────
    print("\n[4] Classical quantum walk refinement (N=50 subgraph)")
    try:
        from quantum_walk import quantum_walk_top_k
        import numpy as np
    except ImportError as e:
        print(f"    SKIPPED (scipy/numpy required: {e})")
    else:
        true_neighbors = build_true_neighbors(edges)
        random.seed(0)
        query_candidates = [n for n in nodes if len(true_neighbors[n['id']]) >= 2]
        random.shuffle(query_candidates)
        query_nodes = query_candidates[:20]

        rrs_qw = []
        recall5_qw = []
        t0 = time.perf_counter()
        for qnode in query_nodes:
            qid = qnode['id']
            true_nbrs = true_neighbors[qid]

            results = quantum_walk_top_k(qid, neighbors_map, edges, k=100, subgraph_size=50)
            ranked_ids = [nid for _, nid in results]

            rr = 0.0
            for rank, nid in enumerate(ranked_ids, 1):
                if nid in true_nbrs:
                    rr = 1.0 / rank
                    break
            rrs_qw.append(rr)

            top5 = set(ranked_ids[:5])
            r5 = len(top5 & true_nbrs) / len(true_nbrs) if true_nbrs else 0.0
            recall5_qw.append(r5)

        t_qw = (time.perf_counter() - t0) / len(query_nodes)
        mrr_qw = sum(rrs_qw) / len(rrs_qw) if rrs_qw else 0.0
        r5_qw = sum(recall5_qw) / len(recall5_qw) if recall5_qw else 0.0
        print(f"    MRR = {mrr_qw:.3f}   Recall@5 = {r5_qw:.3f}")
        print(f"    Avg latency: {t_qw*1000:.1f} ms/query")

    # ── Experiment 5: XOR binding verification ───────────────────────
    print("\n[5] XOR binding (compositional queries) — exactness verification")
    n_trials = 1000
    random.seed(42)
    errors = 0
    for _ in range(n_trials):
        a = random.getrandbits(256)
        b = random.getrandbits(256)
        bound = bind(a, b)
        recovered = unbind(bound, a)
        if recovered != b:
            errors += 1
    print(f"    {n_trials} trials, {errors} bit errors "
          f"(expected: 0 — exact recovery property)")

    # ── Hamming throughput ────────────────────────────────────────────
    print("\n[6] Hamming distance throughput (256-bit)")
    a = random.getrandbits(256)
    b = random.getrandbits(256)
    N = 100000
    t0 = time.perf_counter()
    for _ in range(N):
        hamming(a, b)
    dt = time.perf_counter() - t0
    print(f"    {N/dt/1e6:.2f}M ops/sec ({dt/N*1e6:.2f} µs per comparison)")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY — Reproduced results from paper")
    print("=" * 65)
    print(f"  TA-SDM MRR (target: 0.919): {mrr_ta:.3f}  "
          f"{'✓' if 0.85 < mrr_ta < 0.95 else '≠'}")
    print(f"  TA-SDM Recall@5 (target: 0.652): {r5_ta:.3f}  "
          f"{'✓' if 0.55 < r5_ta < 0.75 else '≠'}")
    print(f"  Content-only MRR (target: 0.35): {mrr_content:.3f}  "
          f"{'✓' if 0.25 < mrr_content < 0.45 else '≠'}")
    if 'mrr_qw' in dir():
        print(f"  Quantum walk MRR @ N=50 (target: 1.000): {mrr_qw:.3f}  "
              f"{'✓' if mrr_qw > 0.95 else '≠'}")
    print(f"  XOR binding errors (target: 0): {errors}  "
          f"{'✓' if errors == 0 else '≠'}")
    print()
    print("  Paper: https://doi.org/10.5281/zenodo.19645323")


if __name__ == '__main__':
    run_benchmark()
