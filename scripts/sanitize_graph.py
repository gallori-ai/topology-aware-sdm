#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanitize_graph.py — strips IP-sensitive content from CEI knowledge graph for public release.

Usage (from repo root):
    python3 scripts/sanitize_graph.py

Input (PRIVATE — never commit):
    data/graph.jsonl   — full nodes with hypothesis text, bisociation, DVU scores
    data/edges.jsonl   — edge list (source/target/type + metadata)

Output (safe to commit):
    data/graph.jsonl   — sanitized nodes: id + type + safe_label + status
    data/edges.jsonl   — sanitized edges: source + target + type only
    data/graph_private_backup.jsonl  — renamed original (gitignored)
    data/edges_private_backup.jsonl  — renamed original (gitignored)

Sanitization strategy:
    KEEP:   id, type, status, safe_label (short name or topic code — no hypothesis text)
    REMOVE: title (full hypothesis), bisociation, falsifiable_test, dvu_estimate,
            doi_urgency, ip_owner, ip_model, ip, tags, connections, file,
            session_id, inherited_from, timestamps, github_issue
    EDGES:  keep source + target + type; strip timestamp, inherited_from, session
    LABEL:  if node has 'label' field → use it (already a short name, not hypothesis)
            if node has 'title' only → use id as label (opaque, not sensitive)
            if node has 'chain' → prepend chain topic to label for clustering

The algorithm (topology_aware_sdm.py) clusters nodes by SimHash(label).
After sanitization, same-topic nodes still cluster together (same chain → same prefix).
Exact MRR numbers will differ; topology-aware improvement direction is preserved.
"""

import json
import shutil
from pathlib import Path


CHAIN_SAFE_LABELS = {
    'quantum_biology': 'quantum biology',
    'quantum_biology_H1': 'quantum biology photosynthesis',
    'quantum_biology_H2': 'quantum biology coherence',
    'quantum_biology_H3': 'quantum biology algorithm',
    'quantum_biology_H4': 'quantum biology density',
    'quantum_biology_H5': 'quantum biology thermoregulation',
    'quantum_biology_H6': 'quantum biology energy computing',
    'quantum_biology_H7': 'quantum biology sensor',
    'quantum_biology_H8': 'quantum biology evolution',
    'quantum_biology_H9': 'quantum biology communication',
    'quantum_biology_H10': 'quantum biology navigation',
}

STATUS_COARSE = {
    'ready': 'ready',
    'researched': 'researched',
    'active': 'active',
    'hypothesis': 'hypothesis',
    'draft': 'draft',
    'registered': 'registered',
    'archived': 'archived',
    'gap': 'gap',
    'open': 'open',
    'partial': 'partial',
    'hold': 'hold',
}


def safe_label(node: dict) -> str:
    """
    Derive a safe, non-IP label for SimHash clustering.

    Priority:
    1. If chain is known: use safe topic name (preserves semantic clustering)
    2. If 'label' exists (v0 style short name): use it directly
    3. Fall back to id (opaque, no hypothesis text)
    """
    chain = node.get('chain', '')
    if chain and chain in CHAIN_SAFE_LABELS:
        return CHAIN_SAFE_LABELS[chain]

    label = node.get('label', '')
    if label and label != node.get('id', ''):
        label = label.replace('â€”', '-').replace('—', '-')
        return label[:80]

    return node.get('id', 'unknown')


def sanitize_node(node: dict) -> dict:
    out = {
        'id': node['id'],
        'type': node.get('type', 'unknown'),
        'label': safe_label(node),
        'status': STATUS_COARSE.get(node.get('status', ''), node.get('status', '')),
    }
    if node.get('chain') and node['chain'] in CHAIN_SAFE_LABELS:
        out['cluster'] = node['chain']
    return out


def sanitize_edge(edge: dict) -> dict:
    return {
        'source': edge.get('source', edge.get('src', '')),
        'target': edge.get('target', edge.get('tgt', '')),
        'type': edge.get('type', ''),
    }


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def write_jsonl(records: list, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main() -> None:
    data_dir = Path('data')
    graph_path = data_dir / 'graph.jsonl'
    edges_path = data_dir / 'edges.jsonl'
    graph_backup = data_dir / 'graph_private_backup.jsonl'
    edges_backup = data_dir / 'edges_private_backup.jsonl'

    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found")
        return

    nodes = load_jsonl(graph_path)
    edges = load_jsonl(edges_path) if edges_path.exists() else []
    print(f"Loaded: {len(nodes)} nodes, {len(edges)} edges")

    # Count sensitive fields before stripping
    has_title = sum(1 for n in nodes if n.get('title'))
    has_bisociation = sum(1 for n in nodes if n.get('bisociation'))
    has_doi_urgency = sum(1 for n in nodes if n.get('doi_urgency'))
    has_ip = sum(1 for n in nodes if n.get('ip') or n.get('ip_owner'))
    print(f"Sensitive fields to strip: title={has_title} bisociation={has_bisociation} "
          f"doi_urgency={has_doi_urgency} ip_fields={has_ip}")

    # Backup originals
    if not graph_backup.exists():
        shutil.copy2(graph_path, graph_backup)
        print(f"Backup created: {graph_backup}")
    if edges_path.exists() and not edges_backup.exists():
        shutil.copy2(edges_path, edges_backup)
        print(f"Backup created: {edges_backup}")

    # Sanitize
    sanitized_nodes = [sanitize_node(n) for n in nodes]
    sanitized_edges = [sanitize_edge(e) for e in edges]

    # Verify edge integrity
    node_ids = {n['id'] for n in sanitized_nodes}
    missing_endpoints = set()
    for e in sanitized_edges:
        if e['source'] not in node_ids:
            missing_endpoints.add(e['source'])
        if e['target'] not in node_ids:
            missing_endpoints.add(e['target'])
    if missing_endpoints:
        print(f"WARNING: {len(missing_endpoints)} edge endpoints not in node set: "
              f"{sorted(missing_endpoints)[:5]}")
    else:
        print(f"Edge integrity: OK (all {len(sanitized_edges)} edges reference valid nodes)")

    # Overwrite with sanitized versions
    write_jsonl(sanitized_nodes, graph_path)
    write_jsonl(sanitized_edges, edges_path)
    print(f"Written: {graph_path} ({len(sanitized_nodes)} nodes)")
    print(f"Written: {edges_path} ({len(sanitized_edges)} edges)")

    # Verify: no sensitive content in output
    with open(graph_path, encoding='utf-8') as f:
        content = f.read()
    sensitive_patterns = ['bisociation', 'falsifiable_test', 'doi_urgency',
                          'dvu_estimate', 'ip_owner', 'patent-candidate', 'Thermal-Substrate']
    found = [p for p in sensitive_patterns if p in content]
    if found:
        print(f"ERROR: sensitive patterns still in output: {found}")
    else:
        print("Verification: no sensitive patterns in sanitized output")

    print()
    print("NEXT STEPS:")
    print("1. Add to .gitignore: data/graph_private_backup.jsonl data/edges_private_backup.jsonl")
    print("2. Run: git add data/graph.jsonl data/edges.jsonl")
    print("3. Run experiments to verify algorithm still produces meaningful results")
    print("4. Commit: docs(data): replace raw graph with sanitized public version — IP protection")
    print("5. Update README: note that labels are simplified for IP protection")
    print("6. URGENT: file DISC-331 on Zenodo before making repo public")


if __name__ == '__main__':
    main()
