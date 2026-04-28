# Topology-Aware Binary SDM for Knowledge Graph Retrieval

**Topology-Aware Binary SDM for Knowledge Graph Retrieval: A Multi-Architecture Empirical Study with Neural Baseline and Quantum Walk Analysis**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19645323.svg)](https://doi.org/10.5281/zenodo.19645323)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/Paper%20License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Author:** Cleber Barcelos Costa (Gallori AI) · [ORCID 0009-0000-5172-9019](https://orcid.org/0009-0000-5172-9019)

---

## Key results at a glance

On a 392-node heterogeneous typed knowledge graph:

| Method | MRR | Recall@5 | Storage |
|--------|-----|----------|---------|
| Content-only SimHash (baseline) | 0.353 | 0.133 | 32 B/node |
| **Topology-Aware SDM (ours)** | **0.919** | **0.652** | **32 B/node** |
| **TA-SDM + classical quantum walk refinement** | **1.000** | **0.753** | 32 B + O(M²) |

- **2.61× MRR improvement** over content-only hashing (full graph)
- **Near-perfect first-rank retrieval** (MRR=1.000) on 50-node *local* subgraphs via classical quantum walk
- **Zero** neural training, GPU, embedding API, or quantum hardware required
- **Identical MRR** reproduced on two CPU generations (Sandy Bridge 2011, Tiger Lake 2020)
- **192× smaller** than 1536-dim float32 neural embeddings
- **Python standard library** only for the core algorithm

## Quick start

```bash
git clone https://github.com/gallori-ai/topology-aware-sdm
cd topology-aware-sdm
pip install -r requirements.txt
python code/benchmark.py
```

Total reproduction time: 20-30 minutes on a modern laptop.

## What this is

A hybrid retrieval method that combines three techniques previously disconnected in the literature:

1. **SimHash content addressing** (Charikar 2002) — locality-sensitive binary hashing of node content
2. **Topology-aware weighted majority vote** over 1-hop graph neighbor signatures — the novel combination
3. **Classical simulation of continuous-time quantum walks** (Farhi & Gutmann 1998) for high-precision subgraph refinement

The full algorithm is under 200 lines of Python, fits in `code/topology_aware_sdm.py`, and uses no external dependencies beyond `numpy` and `scipy` for the quantum walk module.

## What makes it different

- **Exact compositional queries.** `unbind(bind(A, B), A) = B` with zero bit errors — a capability absent from float-embedding systems.
- **Hardware-aligned.** Uses CPU cache hierarchy and hardware POPCNT instruction (via Python's `int.bit_count()`) — no special hardware needed.
- **Training-free.** No gradient descent, no neural model, no data labeling. The address is computed in closed form from content + graph topology.
- **Multi-architecture validated.** MRR = 0.919 reproduced identically on a 2011 Sandy Bridge laptop and a 2020 Tiger Lake laptop.

## Repository structure

```
topology-aware-sdm/
├── README.md                 ← you are here
├── LICENSE                   ← MIT (code) + CC-BY-4.0 (paper)
├── CITATION.cff              ← citation metadata (GitHub-native)
├── .zenodo.json              ← Zenodo deposit metadata
├── requirements.txt          ← numpy, scipy, py-cpuinfo
├── paper/
│   ├── paper-en.md           ← full paper (English)
│   ├── paper-pt.md           ← full paper (Portuguese)
│   └── references.bib        ← bibliography
├── code/
│   ├── topology_aware_sdm.py ← core algorithm (standalone)
│   ├── quantum_walk.py       ← CTQW refinement (scipy-based)
│   └── benchmark.py          ← reproduces all paper results
├── data/
│   ├── graph.jsonl           ← 392-node heterogeneous knowledge graph
│   ├── edges.jsonl           ← 645 typed edges
│   ├── measurements-m1.csv   ← Machine 1 (Tiger Lake) measurements
│   ├── measurements-m2.csv   ← Machine 2 (Sandy Bridge) measurements
│   ├── environment-m1.csv    ← Machine 1 hardware spec
│   └── environment-m2.csv    ← Machine 2 hardware spec
└── docs/
    ├── reproduction.md       ← step-by-step reproduction guide
    └── hardware-comparison.md ← multi-architecture findings
```

## How to cite

If you use this work, please cite:

```bibtex
@misc{barceloscosta2026tasdm,
  author       = {Barcelos Costa, Cleber},
  title        = {{Perfect Knowledge Graph Retrieval via Hybrid
                  Binary SDM and Classical Quantum Walk: A Multi-
                  Architecture Empirical Study}},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19645323},
  url          = {https://doi.org/10.5281/zenodo.19645323}
}
```

See `CITATION.cff` for GitHub-native citation metadata.

## Quick API examples

### Basic retrieval

```python
from topology_aware_sdm import build_addresses, simhash, top_k, load_graph

nodes, edges = load_graph('data/graph.jsonl', 'data/edges.jsonl')
addresses = build_addresses(nodes, edges, bits=256)

# Find nodes near a text query
query_addr = simhash("quantum biology", bits=256)
results = top_k(query_addr, addresses, k=10)
for distance, node_id in results:
    print(f"  {node_id} (hamming={distance})")
```

### Compositional queries (XOR binding)

```python
from topology_aware_sdm import query_and

# Exact conjunction: find nodes related to BOTH topics
results = query_and("quantum physics", "business model", addresses, k=10)
```

### Classical quantum walk refinement

```python
from quantum_walk import quantum_walk_top_k
from collections import defaultdict

neighbors = defaultdict(list)
for e in edges:
    neighbors[e['source']].append(e['target'])
    neighbors[e['target']].append(e['source'])

# Perfect retrieval on 50-node subgraph
results = quantum_walk_top_k('DISC-374', neighbors, edges, k=10, subgraph_size=50)
```

## Reproducing the paper results

```bash
# Basic reproduction on your machine:
python code/benchmark.py

# With hardware fingerprinting (to compare with paper's machines):
pip install py-cpuinfo
python -c "import cpuinfo; print(cpuinfo.get_cpu_info()['brand_raw'])"

# Individual experiments:
python code/topology_aware_sdm.py --query "quantum" --k 10
python code/quantum_walk.py --query-id DISC-374 --subgraph-size 50
```

Expected output: the benchmark ends with a summary showing
`MRR=0.919` (TA-SDM) and `MRR=1.000` (quantum walk at N=50).

## Hardware used for the paper measurements

| | Machine 1 | Machine 2 |
|-|-----------|-----------|
| CPU | Intel Core i7-1165G7 (Tiger Lake, 2020) | Intel Core i7-2637M (Sandy Bridge, 2011) |
| Cores | 4c / 8t | 2c / 4t |
| L3 cache | 12 MB | 4 MB |
| RAM | 16 GB DDR4-3200 | 12 GB DDR3-1333 |
| AVX-512F | yes | no |
| AVX-512 VPOPCNTDQ | **no** | no |
| POPCNT | yes | yes |
| Python | 3.13.5 | 3.14.4 |
| OS | Windows 11 | Windows 10 |

**Nine years** of CPU architectural evolution separates these two machines. The TA-SDM MRR is **identical** (0.919) on both.

## License

- **Code** (`code/`, `requirements.txt`, `CITATION.cff`, `.zenodo.json`): MIT License
- **Paper and written content** (`paper/`, `README.md`, `docs/`): Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Data** (`data/*.jsonl`, `data/*.csv`): CC-BY-4.0

See `LICENSE` for full terms.

## Contact

- **Author:** Cleber Barcelos Costa
- **Affiliation:** Gallori AI (Betim, Minas Gerais, Brazil)
- **ORCID:** [0009-0000-5172-9019](https://orcid.org/0009-0000-5172-9019)
- **Email:** (via Gallori AI)

## Acknowledgments

Multi-machine hardware reproduction was performed via a stigmergic coordination pattern between two instances of an autonomous agent sharing only the Git repository. No direct instance-to-instance communication occurred.
