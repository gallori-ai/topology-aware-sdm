#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Topology-Aware Sparse Distributed Memory (TA-SDM) — standalone implementation.

Paper: "Perfect Knowledge Graph Retrieval via Hybrid Binary SDM and Classical
        Quantum Walk: A Multi-Architecture Empirical Study"
Author: Cleber Barcelos Costa (Gallori AI), ORCID 0009-0000-5172-9019
DOI: 10.5281/zenodo.19645323
License: MIT

Zero external dependencies (Python stdlib only).
Requires: Python 3.10+ for int.bit_count().
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path


# ─── Core primitives ──────────────────────────────────────────────────────

def simhash(text: str, bits: int = 256) -> int:
    """
    Locality-sensitive hash (Charikar 2002).
    Similar text produces small Hamming distance.
    """
    if not text:
        return 0
    words = text.lower().split()
    votes = [0] * bits
    for word in words:
        for seed in (b'h1', b'h2', b'h3', b'h4'):
            h = int.from_bytes(hashlib.sha256(seed + word.encode()).digest(), 'big')
            for b in range(bits):
                votes[b] += 1 if (h >> (b % 256)) & 1 else -1
    result = 0
    for b in range(bits):
        if votes[b] > 0:
            result |= (1 << b)
    return result


def hamming(a: int, b: int) -> int:
    """
    Hamming distance via Python 3.10+ int.bit_count().
    Internally uses hardware POPCNT instruction.
    """
    return (a ^ b).bit_count()


def bind(a: int, b: int) -> int:
    """XOR binding for compositional queries. Invertible, commutative."""
    return a ^ b


def unbind(bound: int, key: int) -> int:
    """Recover operand from bound pair. Exact, zero error."""
    return bound ^ key


# ─── Content extraction ───────────────────────────────────────────────────

def node_to_text(node: dict) -> str:
    """Extract searchable text from a node JSON record."""
    parts = [
        node.get('title') or node.get('label') or '',
        node.get('cluster', ''),
        (node.get('description') or '')[:100],
    ]
    bio = node.get('bisociation', {})
    if isinstance(bio, dict):
        parts.append(bio.get('domain_a', ''))
        parts.append(bio.get('domain_b', ''))
    ft = node.get('falsifiable_test', '')
    if ft:
        parts.append(str(ft)[:80])
    return ' '.join(p for p in parts if p)


# ─── TA-SDM main algorithm ────────────────────────────────────────────────

def content_address(node: dict, bits: int = 256) -> int:
    """Step 1: Pure content SimHash address (before topology enrichment)."""
    return simhash(node_to_text(node), bits)


def topology_address(base: int, neighbor_addrs: list[int], bits: int = 256) -> int:
    """
    Step 2: Topology-aware address via weighted majority vote.

    addr(v) = majority_vote(
        2 × SimHash(content(v)),
        Σ_{u ∈ N(v)} SimHash(content(u))
    )

    Content gets weight 2; each 1-hop neighbor gets weight 1.
    Threshold = (|N(v)| + 2) / 2.

    Returns 256-bit (or bits-bit) binary address as Python int.
    """
    if not neighbor_addrs:
        return base
    votes = [(base >> b & 1) * 2 for b in range(bits)]
    for na in neighbor_addrs:
        for b in range(bits):
            votes[b] += (na >> b & 1)
    threshold = (len(neighbor_addrs) + 2) / 2
    return sum(1 << b for b in range(bits) if votes[b] >= threshold)


def build_addresses(
    nodes: list[dict],
    edges: list[dict],
    bits: int = 256,
) -> dict[str, int]:
    """
    Build TA-SDM addresses for all nodes in the graph.

    Two-pass algorithm:
    Pass 1: compute content-only addresses for all nodes.
    Pass 2: compute topology-aware addresses using pass 1 as neighbor addresses.

    Returns: dict mapping node_id -> 256-bit binary address (as Python int).
    """
    # Build neighbor map (undirected)
    neighbors = defaultdict(list)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src and tgt:
            neighbors[src].append(tgt)
            neighbors[tgt].append(src)

    # Pass 1: content addresses
    nodes_by_id = {n['id']: n for n in nodes}
    content_addrs: dict[str, int] = {}
    for nid, node in nodes_by_id.items():
        content_addrs[nid] = content_address(node, bits)

    # Pass 2: topology-aware addresses
    topo_addrs: dict[str, int] = {}
    for nid, node in nodes_by_id.items():
        base = content_addrs[nid]
        neighbor_ids = neighbors.get(nid, [])
        neighbor_addrs_list = [
            content_addrs[nbr] for nbr in neighbor_ids if nbr in content_addrs
        ]
        topo_addrs[nid] = topology_address(base, neighbor_addrs_list, bits)

    return topo_addrs


# ─── Retrieval ────────────────────────────────────────────────────────────

def rank_by_hamming(
    query: int,
    addresses: dict[str, int],
) -> list[tuple[int, str]]:
    """
    Rank all nodes by Hamming distance to query address.

    Returns: list of (distance, node_id) sorted ascending by distance.
    """
    return sorted(
        ((hamming(query, addr), nid) for nid, addr in addresses.items()),
        key=lambda x: x[0],
    )


def top_k(
    query: int,
    addresses: dict[str, int],
    k: int = 10,
    exclude: set | None = None,
) -> list[tuple[int, str]]:
    """Return top-K nearest nodes by Hamming distance, optionally excluding IDs."""
    exclude = exclude or set()
    ranked = [(d, nid) for d, nid in rank_by_hamming(query, addresses) if nid not in exclude]
    return ranked[:k]


# ─── Compositional queries ────────────────────────────────────────────────

def query_and(
    topic_a: str,
    topic_b: str,
    addresses: dict[str, int],
    bits: int = 256,
    k: int = 10,
) -> list[tuple[int, str]]:
    """
    Compositional AND query via XOR binding.

    Returns nodes nearest to bind(SimHash(A), SimHash(B)) in Hamming space.
    """
    a = simhash(topic_a, bits)
    b = simhash(topic_b, bits)
    bound = bind(a, b)
    return top_k(bound, addresses, k=k)


# ─── I/O helpers ──────────────────────────────────────────────────────────

def load_graph(nodes_path: str | Path, edges_path: str | Path) -> tuple[list, list]:
    """Load nodes and edges from JSONL files."""
    nodes, edges = [], []
    with open(nodes_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    nodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    with open(edges_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    edges.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return nodes, edges


# ─── CLI entry point ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='TA-SDM: Topology-Aware SDM for graph retrieval')
    parser.add_argument('--nodes', default='data/graph.jsonl',
                        help='Path to nodes JSONL (default: data/graph.jsonl)')
    parser.add_argument('--edges', default='data/edges.jsonl',
                        help='Path to edges JSONL (default: data/edges.jsonl)')
    parser.add_argument('--bits', type=int, default=256,
                        help='Address dimension in bits (default: 256)')
    parser.add_argument('--query', type=str, help='Query topic (free text)')
    parser.add_argument('--query-and', type=str, nargs=2, metavar=('A', 'B'),
                        help='Compositional AND query: two topics')
    parser.add_argument('--k', type=int, default=10, help='Top-K results (default: 10)')
    args = parser.parse_args()

    nodes, edges = load_graph(args.nodes, args.edges)
    print(f"[TA-SDM] Loaded {len(nodes)} nodes, {len(edges)} edges")
    print(f"[TA-SDM] Building {args.bits}-bit topology-aware addresses...")
    addresses = build_addresses(nodes, edges, bits=args.bits)
    print(f"[TA-SDM] Built {len(addresses)} addresses.")

    if args.query_and:
        a, b = args.query_and
        print(f"\n[TA-SDM] Compositional query: '{a}' AND '{b}'")
        results = query_and(a, b, addresses, bits=args.bits, k=args.k)
        for i, (dist, nid) in enumerate(results, 1):
            node = next((n for n in nodes if n['id'] == nid), {})
            title = node.get('title') or node.get('label') or ''
            print(f"  {i:2d}. {nid:20s} (hamming={dist:3d})  {title[:60]}")
    elif args.query:
        print(f"\n[TA-SDM] Query: '{args.query}'")
        q = simhash(args.query, args.bits)
        results = top_k(q, addresses, k=args.k)
        for i, (dist, nid) in enumerate(results, 1):
            node = next((n for n in nodes if n['id'] == nid), {})
            title = node.get('title') or node.get('label') or ''
            print(f"  {i:2d}. {nid:20s} (hamming={dist:3d})  {title[:60]}")
    else:
        print("\n[TA-SDM] No query specified. Use --query TEXT or --query-and A B")
        print("\nSample retrieval for node DISCOVERY-083 (if present):")
        if 'DISCOVERY-083' in addresses:
            results = top_k(addresses['DISCOVERY-083'], addresses, k=args.k,
                            exclude={'DISCOVERY-083'})
            for i, (dist, nid) in enumerate(results, 1):
                print(f"  {i:2d}. {nid:20s} (hamming={dist})")
