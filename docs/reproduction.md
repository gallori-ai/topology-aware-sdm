# Reproduction Guide

This document provides step-by-step instructions to reproduce the results in
the paper "Perfect Knowledge Graph Retrieval via Hybrid Binary SDM and
Classical Quantum Walk" (DOI: 10.5281/zenodo.19645323).

## Prerequisites

- Python 3.10 or newer (for `int.bit_count()`)
- 2 GB of RAM minimum
- Any modern x86-64 or ARM64 CPU
- ~15 minutes to 2 hours (depending on hardware generation)

## Setup

```bash
git clone https://github.com/gallori-ai/topology-aware-sdm
cd topology-aware-sdm
pip install -r requirements.txt
```

## Full benchmark (reproduces all paper results)

```bash
python code/benchmark.py
```

Expected output (final summary):

```
SUMMARY — Reproduced results from paper
=====================================================================
  TA-SDM MRR (target: 0.919): 0.919  ✓
  TA-SDM Recall@5 (target: 0.652): 0.652  ✓
  Content-only MRR (target: 0.35): 0.353  ✓
  Quantum walk MRR @ N=50 (target: 1.000): 1.000  ✓
  XOR binding errors (target: 0): 0  ✓
```

MRR values within ±0.01 of these targets are considered reproduced.

## Individual experiments

### TA-SDM retrieval

```bash
# Generic query
python code/topology_aware_sdm.py --query "quantum biology" --k 10

# Compositional AND query (XOR binding)
python code/topology_aware_sdm.py --query-and "quantum" "business" --k 10

# With different bit widths
python code/topology_aware_sdm.py --bits 128 --query "neural" --k 5
python code/topology_aware_sdm.py --bits 1024 --query "neural" --k 5
```

### Quantum walk refinement

```bash
# Classical CTQW on a 50-node BFS subgraph around a query
python code/quantum_walk.py --query-id DISC-374 --subgraph-size 50 --k 10

# Test different evolution times
python code/quantum_walk.py --query-id DISC-374 --t 0.5
python code/quantum_walk.py --query-id DISC-374 --t 1.0
python code/quantum_walk.py --query-id DISC-374 --t 2.0
```

### Hybrid retrieval (TA-SDM + CTQW)

```python
from topology_aware_sdm import build_addresses, load_graph
from quantum_walk import hybrid_retrieval
from collections import defaultdict

nodes, edges = load_graph('data/graph.jsonl', 'data/edges.jsonl')
addresses = build_addresses(nodes, edges, bits=256)

neighbors = defaultdict(list)
for e in edges:
    neighbors[e['source']].append(e['target'])
    neighbors[e['target']].append(e['source'])

results = hybrid_retrieval(
    query_id='DISC-374',
    neighbors_map=neighbors,
    edges=edges,
    ta_sdm_addresses=addresses,
    k=10,
    refinement_threshold_k=50,
)

for source, node_id in results:
    print(f"  [{source}] {node_id}")
```

## Hardware fingerprinting

To capture your machine's specifications for comparison with the paper's
machines:

```bash
pip install py-cpuinfo
python -c "
import cpuinfo, platform
info = cpuinfo.get_cpu_info()
print(f'CPU: {info[\"brand_raw\"]}')
print(f'Cores: {info[\"count\"]}')
print(f'L3 cache: {info.get(\"l3_cache_size\", \"unknown\")} bytes')
print(f'POPCNT: {\"popcnt\" in info[\"flags\"]}')
print(f'AVX2: {\"avx2\" in info[\"flags\"]}')
print(f'AVX-512F: {\"avx512f\" in info[\"flags\"]}')
print(f'AVX-512 VPOPCNTDQ: {\"avx512_vpopcntdq\" in info[\"flags\"]}')
print(f'OS: {platform.platform()}')
print(f'Python: {platform.python_version()}')
"
```

## Expected variance across hardware

The paper validates reproducibility on two specific machines:
- Dell Vostro 5402 (Intel Core i7-1165G7, Tiger Lake, 2020)
- LG Z430 (Intel Core i7-2637M, Sandy Bridge, 2011)

On both machines:
- **MRR** values are identical to 3 decimal places (algorithm property)
- **Throughput** varies with CPU generation: expect ~4-5x difference between
  consumer laptops of different generations
- **Latency** scales with throughput inversely

If you see MRR values outside the range 0.85–0.95 for TA-SDM, please:
1. Check that `python --version` is 3.10+
2. Verify the graph files have the expected node count (392)
3. Compare your hardware specs — some combinations may have issues with
   Python's big-integer arithmetic

## Troubleshooting

**"ModuleNotFoundError: No module named 'scipy'"**
→ Run `pip install -r requirements.txt`. The quantum walk module requires scipy.

**"AttributeError: 'int' object has no attribute 'bit_count'"**
→ You're on Python < 3.10. Upgrade to 3.10 or newer. `bit_count()` was added in
3.10 for fast POPCNT.

**MRR ≈ 0.05 (random) instead of 0.919**
→ Check that `build_addresses()` is being called with both nodes AND edges.
Without edges, the topology aggregation fails silently and addresses become
content-only.

**Very slow execution (> 10 minutes for benchmark on modern hardware)**
→ Your Python int arithmetic may not be using POPCNT hardware. Check with
`py-cpuinfo` that your CPU flags include `popcnt`. Very old CPUs (pre-2008)
lack POPCNT and fall back to software.

## Citation

```bibtex
@misc{barceloscosta2026tasdm,
  author = {Barcelos Costa, Cleber},
  title  = {Perfect Knowledge Graph Retrieval via Hybrid Binary SDM
            and Classical Quantum Walk: A Multi-Architecture Empirical Study},
  year   = {2026},
  doi    = {10.5281/zenodo.19645323},
  url    = {https://doi.org/10.5281/zenodo.19645323}
}
```

## Contact

Bug reports or questions: please open a GitHub issue at
https://github.com/gallori-ai/topology-aware-sdm/issues
