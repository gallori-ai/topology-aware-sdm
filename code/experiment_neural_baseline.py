#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C1 - Sentence-Transformers Neural Baseline
Reviewer Opus 4.7: "Add baseline with sentence-transformers/all-MiniLM-L6-v2
(384d, open-source, no API) running on the same 392 nodes."

Compares head-to-head on the same graph:
  - all-MiniLM-L6-v2 (384d float32 neural embedding)
  - TA-SDM (256-bit binary, ours)
  - Content-only SimHash (256-bit binary baseline)

Multi-seed statistical protocol.
"""

import sys
import time
import random
import statistics
import numpy as np
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from topology_aware_sdm import load_graph, node_to_text, simhash, build_addresses

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def compute_neural_embeddings(nodes):
    """Compute 384-dim float32 embeddings via sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print("  Loading all-MiniLM-L6-v2 (may download ~90MB on first run)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Model loaded. Dimension: {model.get_sentence_embedding_dimension()}")

    texts = [node_to_text(n) or '(empty)' for n in nodes]
    print(f"  Encoding {len(texts)} nodes...")
    t0 = time.perf_counter()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    t_encode = time.perf_counter() - t0
    print(f"  Encoded in {t_encode:.1f}s ({t_encode/len(texts)*1000:.1f}ms per node)")

    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return {n['id']: embeddings[i] for i, n in enumerate(nodes)}


def cosine_rank(query_vec, all_vecs_dict, query_id):
    """Rank nodes by cosine similarity to query (highest first)."""
    ranked = []
    for nid, vec in all_vecs_dict.items():
        if nid == query_id:
            continue
        sim = float(np.dot(query_vec, vec))
        ranked.append((-sim, nid))  # negative for descending sort
    ranked.sort()
    return [nid for _, nid in ranked]


def hamming_rank(query_addr, addresses, query_id):
    """Rank nodes by Hamming distance (closest first)."""
    ranked = []
    for nid, addr in addresses.items():
        if nid == query_id:
            continue
        d = (query_addr ^ addr).bit_count()
        ranked.append((d, nid))
    ranked.sort()
    return [nid for _, nid in ranked]


def measure_mrr(ranked_ids, true_neighbors):
    rr = 0.0
    for rank, nid in enumerate(ranked_ids, 1):
        if nid in true_neighbors:
            rr = 1.0 / rank
            break
    top5 = set(ranked_ids[:5])
    r5 = len(top5 & true_neighbors) / len(true_neighbors) if true_neighbors else 0.0
    return rr, r5


def run_seed(seed, nodes, edges, neural_vecs, ta_addrs, content_addrs, n_queries=50):
    """Run MRR measurement for one seed."""
    true_neighbors = defaultdict(set)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src and tgt:
            true_neighbors[src].add(tgt)
            true_neighbors[tgt].add(src)

    query_candidates = [n for n in nodes if len(true_neighbors[n['id']]) >= 2]
    random.seed(seed)
    random.shuffle(query_candidates)
    query_nodes = query_candidates[:n_queries]

    results = {'neural': [], 'ta_sdm': [], 'content': []}
    results_r5 = {'neural': [], 'ta_sdm': [], 'content': []}

    for qnode in query_nodes:
        qid = qnode['id']
        true_nbrs = true_neighbors[qid]

        # Neural: cosine similarity
        rank_neural = cosine_rank(neural_vecs[qid], neural_vecs, qid)
        mrr_n, r5_n = measure_mrr(rank_neural, true_nbrs)

        # TA-SDM: Hamming distance
        rank_ta = hamming_rank(ta_addrs[qid], ta_addrs, qid)
        mrr_t, r5_t = measure_mrr(rank_ta, true_nbrs)

        # Content-only
        rank_c = hamming_rank(content_addrs[qid], content_addrs, qid)
        mrr_c, r5_c = measure_mrr(rank_c, true_nbrs)

        results['neural'].append(mrr_n)
        results['ta_sdm'].append(mrr_t)
        results['content'].append(mrr_c)
        results_r5['neural'].append(r5_n)
        results_r5['ta_sdm'].append(r5_t)
        results_r5['content'].append(r5_c)

    return {
        k: {'mrr_mean': statistics.mean(v), 'r5_mean': statistics.mean(results_r5[k])}
        for k, v in results.items()
    }, results, results_r5


def main():
    print("=" * 65)
    print("C1 - Sentence-Transformers Neural Baseline")
    print("Head-to-head: all-MiniLM-L6-v2 (384d float32) vs TA-SDM (256-bit binary)")
    print("=" * 65)

    repo_root = Path(__file__).parent.parent
    nodes, edges = load_graph(
        str(repo_root / 'data' / 'graph.jsonl'),
        str(repo_root / 'data' / 'edges.jsonl'),
    )
    print(f"\nGraph: {len(nodes)} nodes, {len(edges)} edges")

    # ── Compute embeddings and addresses ────────────────────────
    print("\n[1/3] Computing neural embeddings (all-MiniLM-L6-v2)...")
    neural_vecs = compute_neural_embeddings(nodes)

    print("\n[2/3] Computing TA-SDM addresses (256-bit, 1-hop)...")
    t0 = time.perf_counter()
    ta_addrs = build_addresses(nodes, edges, bits=256)
    print(f"  {(time.perf_counter()-t0)*1000:.0f}ms")

    print("\n[3/3] Computing content-only SimHash (256-bit)...")
    t0 = time.perf_counter()
    content_addrs = {n['id']: simhash(node_to_text(n), 256) for n in nodes}
    print(f"  {(time.perf_counter()-t0)*1000:.0f}ms")

    # ── Run 10 seeds ────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("Running 10 seeds (seed=0..9), 50 queries each")
    print("-" * 65)
    print(f"{'Seed':<6} {'Neural MRR':<14} {'TA-SDM MRR':<14} {'Content MRR':<14}")
    print("-" * 65)

    all_neural_mrrs, all_ta_mrrs, all_content_mrrs = [], [], []
    all_neural_r5, all_ta_r5, all_content_r5 = [], [], []

    for seed in range(10):
        _, res, res_r5 = run_seed(seed, nodes, edges, neural_vecs, ta_addrs, content_addrs)

        n_mrr = statistics.mean(res['neural'])
        t_mrr = statistics.mean(res['ta_sdm'])
        c_mrr = statistics.mean(res['content'])

        all_neural_mrrs.append(n_mrr)
        all_ta_mrrs.append(t_mrr)
        all_content_mrrs.append(c_mrr)

        all_neural_r5.append(statistics.mean(res_r5['neural']))
        all_ta_r5.append(statistics.mean(res_r5['ta_sdm']))
        all_content_r5.append(statistics.mean(res_r5['content']))

        print(f"{seed:<6} {n_mrr:<14.4f} {t_mrr:<14.4f} {c_mrr:<14.4f}")

    # ── Statistics ──────────────────────────────────────────────
    def stats(xs):
        m = statistics.mean(xs)
        s = statistics.stdev(xs)
        sem = s / (len(xs) ** 0.5)
        return m, s, m - 1.96 * sem, m + 1.96 * sem

    neural_mrr, neural_std, neural_lo, neural_hi = stats(all_neural_mrrs)
    ta_mrr, ta_std, ta_lo, ta_hi = stats(all_ta_mrrs)
    content_mrr, content_std, _, _ = stats(all_content_mrrs)

    neural_r5, _, _, _ = stats(all_neural_r5)
    ta_r5, _, _, _ = stats(all_ta_r5)

    print("\n" + "=" * 65)
    print("RESULTS — 10-seed statistical comparison")
    print("=" * 65)
    print(f"\n{'Method':<30} {'MRR mean ± std':<25} {'95% CI':<25}")
    print("-" * 65)
    print(f"{'Neural 384d float32':<30} {f'{neural_mrr:.4f} ± {neural_std:.4f}':<25} [{neural_lo:.4f}, {neural_hi:.4f}]")
    print(f"{'TA-SDM 256-bit binary (ours)':<30} {f'{ta_mrr:.4f} ± {ta_std:.4f}':<25} [{ta_lo:.4f}, {ta_hi:.4f}]")
    print(f"{'Content-only 256-bit':<30} {f'{content_mrr:.4f} ± {content_std:.4f}':<25}")

    # Storage comparison
    print(f"\nStorage per node:")
    print(f"  Neural float32 384d: {384 * 4} bytes = 1536 bytes")
    print(f"  TA-SDM binary 256-bit: {256 // 8} bytes = 32 bytes")
    print(f"  Compression ratio: {1536/32:.1f}x smaller for TA-SDM")

    # Paired t-test: TA-SDM vs Neural
    print(f"\nPaired t-test TA-SDM vs Neural:")
    diffs = [t - n for t, n in zip(all_ta_mrrs, all_neural_mrrs)]
    diff_mean = statistics.mean(diffs)
    diff_std = statistics.stdev(diffs)
    if diff_std > 0:
        t_stat = diff_mean / (diff_std / (len(diffs) ** 0.5))
        print(f"  Mean paired difference: {diff_mean:+.4f} (positive = TA-SDM better)")
        print(f"  Std of difference: {diff_std:.4f}")
        print(f"  t-statistic: {t_stat:+.2f} (df=9)")
        if abs(t_stat) > 4.781:
            sig = "p < 0.001 (highly significant)"
        elif abs(t_stat) > 3.250:
            sig = "p < 0.01  (significant)"
        elif abs(t_stat) > 2.262:
            sig = "p < 0.05  (significant)"
        else:
            sig = "p > 0.05  (not significant — methods are statistically tied)"
        print(f"  {sig}")

    # Interpretation
    print("\n" + "=" * 65)
    print("INTERPRETATION FOR PAPER 4 / MAIN PAPER")
    print("=" * 65)
    if ta_mrr > neural_mrr + 0.05:
        print(f"TA-SDM BEATS neural embedding by {ta_mrr - neural_mrr:.3f} MRR at 48x less storage.")
        print("Paper claim: 'Binary TA-SDM exceeds neural embeddings on graph neighbor retrieval'")
    elif neural_mrr > ta_mrr + 0.05:
        print(f"Neural embedding BEATS TA-SDM by {neural_mrr - ta_mrr:.3f} MRR, but uses 48x more storage.")
        print("Paper claim: 'Neural embedding wins on quality; TA-SDM wins on storage+cost'")
    else:
        print(f"TA-SDM and neural are STATISTICALLY CLOSE ({abs(ta_mrr - neural_mrr):.3f} MRR difference).")
        print("Paper claim: 'TA-SDM matches neural embeddings at 48x less storage, no API, no GPU'")

    # Write CSV
    out_csv = repo_root / 'data' / 'neural-baseline-m1.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("seed,neural_mrr,ta_sdm_mrr,content_mrr,neural_r5,ta_sdm_r5,content_r5\n")
        for seed in range(10):
            f.write(f"{seed},{all_neural_mrrs[seed]:.4f},{all_ta_mrrs[seed]:.4f},"
                    f"{all_content_mrrs[seed]:.4f},{all_neural_r5[seed]:.4f},"
                    f"{all_ta_r5[seed]:.4f},{all_content_r5[seed]:.4f}\n")
    try:
        rel = out_csv.relative_to(Path.cwd())
        print(f"\n[CSV] Written to {rel}")
    except ValueError:
        print(f"\n[CSV] Written to {out_csv}")


if __name__ == '__main__':
    main()
