#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 - Multi-seed Statistical Rigor for MRR claims
Reviewer Opus 4.7 request: "Run 10 different seeds and report mean MRR +/- std"

Confidence interval via 10 bootstrap seeds (seed=0..9).
"""

import sys
import time
import statistics
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sdm_benchmark import load_graph, measure_mrr, simhash, node_to_text, hamming
from experiment_2hop import compute_all_addresses

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def measure_mrr_with_seed(nodes, edges, addr_fn, seed: int, n_queries: int = 50):
    """Adapted measure_mrr that uses a specific seed."""
    import random as rnd

    # Build true neighbors
    true_neighbors = defaultdict(set)
    for e in edges:
        src = e.get('source', e.get('src', ''))
        tgt = e.get('target', e.get('tgt', ''))
        if src and tgt:
            true_neighbors[src].add(tgt)
            true_neighbors[tgt].add(src)

    # Select query nodes with this seed
    query_candidates = [n for n in nodes if len(true_neighbors[n['id']]) >= 2]
    rnd.seed(seed)
    rnd.shuffle(query_candidates)
    query_nodes = query_candidates[:n_queries]

    addrs = {n['id']: addr_fn(n) for n in nodes}

    rrs = []
    recall5_scores = []
    for qnode in query_nodes:
        qid = qnode['id']
        qaddr = addrs[qid]
        true_nbrs = true_neighbors[qid]

        ranked = sorted(
            [(hamming(qaddr, addrs[nid]), nid) for nid in addrs if nid != qid]
        )

        rr = 0.0
        for rank, (_, nid) in enumerate(ranked, 1):
            if nid in true_nbrs:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        top5_ids = {nid for _, nid in ranked[:5]}
        r5 = len(top5_ids & true_nbrs) / len(true_nbrs) if true_nbrs else 0.0
        recall5_scores.append(r5)

    return (
        sum(rrs) / len(rrs) if rrs else 0.0,
        sum(recall5_scores) / len(recall5_scores) if recall5_scores else 0.0,
    )


def main():
    print("=" * 65)
    print("C3 - Multi-Seed Statistical Rigor")
    print("10 seeds, 50 queries each -> mean MRR +/- std + 95% CI")
    print("=" * 65)

    nodes, edges = load_graph()
    print(f"\nGraph: {len(nodes)} nodes, {len(edges)} edges")

    # Compute topology-aware addresses once (they don't depend on query seed)
    print("\nComputing TA-SDM addresses (256-bit, 1-hop)...")
    t0 = time.perf_counter()
    topo_addrs = compute_all_addresses(nodes, edges, k=1, bits=256)
    t_index = time.perf_counter() - t0
    print(f"  Indexing: {t_index*1000:.0f}ms")

    # Compute content-only addresses for baseline
    content_addrs = {n['id']: simhash(node_to_text(n), 256) for n in nodes}

    # Run 10 seeds
    print("\n" + "-" * 65)
    print("Running 10 seeds (seed=0..9)...")
    print("-" * 65)
    print(f"{'Seed':<6} {'MRR (TA-SDM)':<15} {'Recall@5 (TA-SDM)':<18} {'MRR (content)':<15}")
    print("-" * 65)

    ta_mrrs = []
    ta_r5s = []
    ct_mrrs = []
    ct_r5s = []
    for seed in range(10):
        mrr_ta, r5_ta = measure_mrr_with_seed(
            nodes, edges, lambda n: topo_addrs[n['id']], seed=seed
        )
        mrr_ct, r5_ct = measure_mrr_with_seed(
            nodes, edges, lambda n: content_addrs[n['id']], seed=seed
        )
        ta_mrrs.append(mrr_ta)
        ta_r5s.append(r5_ta)
        ct_mrrs.append(mrr_ct)
        ct_r5s.append(r5_ct)
        print(f"{seed:<6} {mrr_ta:<15.4f} {r5_ta:<18.4f} {mrr_ct:<15.4f}")

    # Statistics
    def stats(xs):
        mean = statistics.mean(xs)
        stdev = statistics.stdev(xs) if len(xs) > 1 else 0.0
        # 95% CI via t-distribution approximation (for small n use t=2.262 for n=10)
        # Using normal approximation: 1.96 * std / sqrt(n)
        sem = stdev / (len(xs) ** 0.5)
        ci_half = 1.96 * sem
        return mean, stdev, mean - ci_half, mean + ci_half

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    ta_mrr_mean, ta_mrr_std, ta_mrr_lo, ta_mrr_hi = stats(ta_mrrs)
    ta_r5_mean, ta_r5_std, ta_r5_lo, ta_r5_hi = stats(ta_r5s)
    ct_mrr_mean, ct_mrr_std, _, _ = stats(ct_mrrs)
    ct_r5_mean, ct_r5_std, _, _ = stats(ct_r5s)

    print(f"\nTA-SDM (256-bit, 1-hop) over 10 seeds:")
    print(f"  MRR:      mean = {ta_mrr_mean:.4f}")
    print(f"            std  = {ta_mrr_std:.4f}")
    print(f"            95% CI = [{ta_mrr_lo:.4f}, {ta_mrr_hi:.4f}]")
    print(f"  Recall@5: mean = {ta_r5_mean:.4f}")
    print(f"            std  = {ta_r5_std:.4f}")
    print(f"            95% CI = [{ta_r5_lo:.4f}, {ta_r5_hi:.4f}]")

    print(f"\nContent-only baseline over 10 seeds:")
    print(f"  MRR:      mean = {ct_mrr_mean:.4f}, std = {ct_mrr_std:.4f}")
    print(f"  Recall@5: mean = {ct_r5_mean:.4f}, std = {ct_r5_std:.4f}")

    # Improvement ratio
    print(f"\nImprovement ratio (multi-seed):")
    ratio_mrr = ta_mrr_mean / ct_mrr_mean if ct_mrr_mean > 0 else float('inf')
    ratio_r5 = ta_r5_mean / ct_r5_mean if ct_r5_mean > 0 else float('inf')
    print(f"  MRR:      {ratio_mrr:.2f}x")
    print(f"  Recall@5: {ratio_r5:.2f}x")

    # Paired t-test approximation
    print(f"\nPaired t-test (TA-SDM vs content-only MRR):")
    diffs = [ta - ct for ta, ct in zip(ta_mrrs, ct_mrrs)]
    diff_mean = statistics.mean(diffs)
    diff_std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    if diff_std > 0:
        t_stat = diff_mean / (diff_std / (len(diffs) ** 0.5))
        print(f"  Mean difference: {diff_mean:.4f}")
        print(f"  Std of difference: {diff_std:.4f}")
        print(f"  t-statistic: {t_stat:.2f} (df=9)")
        # critical value for df=9, alpha=0.001: ~4.781
        if abs(t_stat) > 4.781:
            print(f"  p < 0.001 (highly significant, critical t=4.781 for df=9)")
        elif abs(t_stat) > 3.250:
            print(f"  p < 0.01  (significant, critical t=3.250 for df=9)")
        elif abs(t_stat) > 2.262:
            print(f"  p < 0.05  (significant, critical t=2.262 for df=9)")
        else:
            print(f"  p > 0.05  (not significant at alpha=0.05)")

    # Write CSV
    from pathlib import Path
    out_csv = Path(__file__).parent.parent.parent / 'knowledge/papers/series-DISC-374/public-release/data/multiseed-m1.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write("seed,mrr_ta_sdm,recall5_ta_sdm,mrr_content,recall5_content\n")
        for seed in range(10):
            f.write(f"{seed},{ta_mrrs[seed]:.4f},{ta_r5s[seed]:.4f},{ct_mrrs[seed]:.4f},{ct_r5s[seed]:.4f}\n")
    print(f"\n[CSV] Written to {out_csv.relative_to(Path.cwd())}")

    return {
        'ta_mrr_mean': ta_mrr_mean,
        'ta_mrr_std': ta_mrr_std,
        'ta_mrr_ci': (ta_mrr_lo, ta_mrr_hi),
        'ct_mrr_mean': ct_mrr_mean,
        'improvement_ratio': ratio_mrr,
    }


if __name__ == '__main__':
    main()
