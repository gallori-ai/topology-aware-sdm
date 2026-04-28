# Systematic Literature Review (Extended) — TA-SDM and Classical Quantum Walk

**Companion to:** Topology-Aware Binary SDM for Knowledge Graph Retrieval:
A Multi-Architecture Empirical Study with Neural Baseline and Quantum Walk Analysis
**Paper DOI:** [10.5281/zenodo.19645323](https://doi.org/10.5281/zenodo.19645323)
**Author:** Cleber Barcelos Costa (Gallori AI)
**Date:** 2026-04-18

---

## 1. Review Methodology

This document expands the literature review summarized in Section 2 of the main
paper. The review was conducted as a structured (not PRISMA-compliant) review by
a single author, targeting five research traditions adjacent to the proposed
method.

### 1.1 Research question

> "What is the state of prior work on methods combining (a) binary content
> hashing, (b) per-node graph topology aggregation, and (c) quantum-walk-inspired
> retrieval, for the task of finding near-neighbors in heterogeneous typed
> knowledge graphs?"

### 1.2 Databases queried

- Google Scholar (via automated scraping, 2026-04)
- Semantic Scholar API
- ACM Digital Library (Library indexed search)
- IEEE Xplore
- arXiv categories: cs.IR, cs.LG, cs.DS, cs.DB, quant-ph

### 1.3 Search terms

Boolean combinations of the following, with publication year filter 1988-2026:

**Core terms:**
- "sparse distributed memory" + "graph"
- "hyperdimensional computing" + "retrieval"
- "SimHash" + "graph neighbor"
- "locality-sensitive hashing" + "graph topology"
- "continuous-time quantum walk" + "retrieval"
- "classical simulation quantum walk" + "graph"

**Auxiliary terms:**
- "binary embeddings" + "knowledge graph"
- "graph neural network" + "fixed weights" + "binary"
- "1-hop aggregation" + "retrieval"
- "XOR binding" + "graph"

### 1.4 Inclusion criteria

IC1. Peer-reviewed publication OR preprint (arXiv, Zenodo) OR authoritative
technical report (e.g., NIST, IBM Research).

IC2. Primary methodology is either:
     (a) theoretical foundation relevant to our 5 traditions, OR
     (b) empirical results on graph retrieval / document retrieval / binary
         embedding.

IC3. Primary language English or Portuguese.

IC4. Published or released between 1988 (Kanerva's SDM foundational work) and
     2026-04-18 (submission date of this work).

### 1.5 Exclusion criteria

EC1. Non-retrieval applications of the methods (e.g., reservoir computing
     applied only to chaotic time series without retrieval evaluation).

EC2. Hardware-only optimization papers without retrieval methodology
     contribution.

EC3. Industrial application reports without methodological novelty.

EC4. Editorials, letters to the editor, book reviews.

### 1.6 Selection process

Initial search returned ≈ 240 candidates. After title and abstract screening,
≈ 120 remained. After full-text screening against inclusion/exclusion criteria,
47 were included in this review. Selection was performed by a single author
(no blind second reviewer), so the review is classified as "structured" rather
than PRISMA-compliant.

---

## 2. Tradition 1 — Sparse Distributed Memory (SDM)

### 2.1 Foundational work

**Kanerva, P. (1988).** *Sparse Distributed Memory.* MIT Press.

Introduces the SDM model: a binary address space of dimension d, N hard
locations at random addresses, writing data at address A stores in activated
set {i : Hamming(A, h_i) ≤ r}, reading retrieves majority vote over activated
set's stored data.

Key theoretical results:
- Capacity scales as N/log(N) items can be stored with retrieval error ≤ ε
- Robust to address noise up to ~20% bit errors
- Biologically plausible as model of episodic memory

### 2.2 SDM theoretical extensions

**Hinton, G. E., & Anderson, J. A. (Eds.). (1989).** *Parallel Models of
Associative Memory.* Psychology Press.

Collection of extensions to SDM including relation to Hopfield networks,
learning rules, and biological plausibility.

**Ramsauer, H., Schäfl, B., Lehner, J., et al. (2020).** Hopfield Networks is
All You Need. *NeurIPS 2020.*

**Key result for our work:** Proves that modern Hopfield networks (a continuous
relaxation of SDM) are mathematically equivalent to the attention mechanism in
transformers:

softmax(QK^T / √d) V ≡ continuous-relaxation SDM retrieval

This equivalence implies that for binary retrieval tasks where continuous
relaxation provides no benefit, SDM at the binary level should be competitive
with transformer attention at fractional cost.

### 2.3 Recent SDM applications

**Imani, M., et al. (2017-2022).** Various papers on hyperdimensional computing
and SDM for neuromorphic systems. These target sensor fusion, robotics, and
embedded systems, not heterogeneous knowledge graph retrieval.

### 2.4 Gap identified

SDM has been extensively studied for:
- Episodic memory and recall tasks
- Sensor fusion
- Pattern completion
- Robotic perception

Not found in the surveyed literature:
- Application to heterogeneous typed knowledge graphs
- Per-node address computation that explicitly incorporates graph topology
- Combination with XOR-bind for compositional retrieval at the node level

---

## 3. Tradition 2 — Hyperdimensional / Vector-Symbolic Computing (HDC / VSA)

### 3.1 Foundational work

**Plate, T. A. (1995).** Holographic Reduced Representations. *IEEE TNN* 6(3).

Introduces circular convolution as binding for distributed representations in
float vectors. Establishes the mathematical basis for vector-symbolic
architectures.

**Plate, T. A. (2003).** *Holographic Reduced Representations.* CSLI Publications.

Book-length treatment. Introduces XOR binding on binary vectors as the
discrete counterpart of circular convolution.

### 3.2 HDC for graphs

**Mitrokhin, A., et al. (2019).** Learning sensorimotor control with
neuromorphic sensors: Toward hyperdimensional active perception.
*Science Robotics* 4(30).

First major application of HDC to graph-like structures. Encodes sensor
trajectories as hypervectors. Applied to robotic perception.

**Poduval, P., Alimohamadi, H., Zakeri, A., & Imani, M. (2022).** GrapHD:
Graph-Based Hyperdimensional Memorization for Brain-Like Cognitive Learning.
*IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems.*

**Most relevant prior work.** Encodes graphs as single hypervectors:

  graph_vec = Σ_{(s,t) ∈ E} bind(src_vec, dst_vec)

Supports graph-level classification (is this graph similar to that graph?) but
not per-node retrieval. The graph_vec cannot be "queried for its nodes."

### 3.3 Other VSA variants

**Eliasmith, C. (2013).** *How to Build a Brain.* Oxford Univ. Press.

Introduces Semantic Pointer Architecture (SPA). Uses circular convolution for
binding at the cognitive level. Not specifically for retrieval.

**Frady, E. P., et al. (2021-2023).** Series of papers on Vector-Symbolic
Binding and Sparse Block Codes. Focus on biological plausibility and
neuromorphic implementation.

### 3.4 Gap identified

HDC graph encoding (GrapHD etc.) operates at whole-graph level:
- One vector represents the entire graph
- Query is "is this graph similar to another graph"
- Individual nodes cannot be retrieved from graph_vec

Our contribution inverts this: we compute a *per-node* binary address that
incorporates HDC-style aggregation (specifically, weighted binary majority
over neighbor signatures) while retaining node-level retrievability.

---

## 4. Tradition 3 — Locality-Sensitive Hashing (LSH)

### 4.1 Foundational work

**Indyk, P., & Motwani, R. (1998).** Approximate Nearest Neighbors: Towards
Removing the Curse of Dimensionality. *STOC 1998.*

Formalizes locality-sensitive hashing. Proves sublinear-time approximate
nearest neighbor search is achievable with proper hash family.

**Charikar, M. (2002).** Similarity estimation techniques from rounding
algorithms. *STOC 2002.*

Introduces SimHash for cosine similarity preservation. Our work uses SimHash
directly as the content address before topology enrichment.

**Broder, A. (1997).** On the resemblance and containment of documents.
*Compression and Complexity of Sequences.*

Introduces MinHash for Jaccard similarity estimation on sets. Complementary to
SimHash; used for set-valued data rather than vector-valued.

### 4.2 LSH applications

**Manku, G. S., Jain, A., & Das Sarma, A. (2007).** Detecting Near-Duplicates
for Web Crawling. *WWW 2007.*

Application of SimHash to web-scale deduplication. Establishes SimHash's
industrial utility.

**Paulevé, L., Jégou, H., & Amsaleg, L. (2010).** Locality sensitive hashing:
A comparison of hash function types and querying mechanisms. *Pattern
Recognition Letters* 31(11).

Empirical comparison of LSH families. Identifies SimHash as strong for token-
based content.

### 4.3 LSH for graphs

**Cao, S., Lu, W., & Xu, Q. (2016).** Deep Neural Networks for Learning Graph
Representations. *AAAI 2016.*

Uses LSH as preliminary step before deep embedding. LSH is auxiliary, not the
primary method.

**Hamilton, W. L., Ying, R., & Leskovec, J. (2017).** Inductive Representation
Learning on Large Graphs. *NeurIPS.*

GraphSAGE uses neighborhood sampling but not LSH-based aggregation. The
neighborhood aggregation is via trained neural layers, not binary hashing.

### 4.4 Gap identified

LSH research has focused on:
- Content-only hashing (SimHash, MinHash)
- Hash-based indexing (LSH forests, multi-probe LSH)
- Industrial deduplication

Not found:
- Explicit incorporation of graph topology into the LSH via neighbor-signature
  majority voting
- Exact compositional queries via XOR on LSH outputs
- Empirical evaluation of topology-enriched LSH against transformer-embedding
  baselines

---

## 5. Tradition 4 — Graph Neural Networks (GNN)

### 5.1 Foundational work

**Kipf, T. N., & Welling, M. (2017).** Semi-Supervised Classification with
Graph Convolutional Networks. *ICLR 2017.*

Introduces GCN: per-layer operation H^(l+1) = σ(D^{-1/2} A D^{-1/2} H^(l) W^(l)).

Trainable weights W per layer enable k-hop aggregation to be selective about
which neighbors to weight.

**Hamilton, W. L., Ying, R., & Leskovec, J. (2017).** Inductive Representation
Learning on Large Graphs. *NeurIPS.*

GraphSAGE generalizes GCN to inductive setting with sampled neighborhoods.

**Veličković, P., et al. (2018).** Graph Attention Networks. *ICLR 2018.*

Adds attention mechanism to neighborhood aggregation.

**Grover, A., & Leskovec, J. (2016).** node2vec: Scalable Feature Learning for
Networks. *KDD 2016.*

Trains node embeddings via biased random walks. Float-valued embeddings.

### 5.2 GNN depth literature

**Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019).** How Powerful Are Graph
Neural Networks? *ICLR.*

Analyzes expressive power of GNNs. Proves that max depth useful is bounded by
the Weisfeiler-Leman test.

**Li, Q., Han, Z., & Wu, X. (2018).** Deeper Insights into Graph Convolutional
Networks for Semi-Supervised Learning. *AAAI.*

Identifies "oversmoothing" problem: GCN at depth >2-3 often degrades
performance because features become too similar across neighbors.

### 5.3 Binary GNNs

**Wang, H., et al. (2021).** Binary Graph Neural Networks. *CVPR.*

Quantizes GCN weights and activations to binary. Still requires training.
Achieves speedup over float GCN. Does not use content-addressing.

### 5.4 Gap identified

GCN-family methods require:
- Training (gradient descent) with labeled supervision
- Float-valued compute (GPU acceleration)
- Hyperparameter search for depth and layer widths

Our method is:
- Training-free (closed-form)
- Binary compute (POPCNT hardware instruction)
- Fixed parameters (no hyperparameter search)

The combination of fixed-weight (identity weights) + binary majority vote +
content-aware aggregation does not appear in the surveyed GNN literature. It
is closest in spirit to "untrained GCN" — a configuration rarely evaluated in
the GNN literature because the field focuses on learned representations.

The "oversmoothing" finding at depth >2-3 provides a theoretical basis for our
empirical observation that 1-hop is optimal in our setting (Section 7.1 of
main paper).

---

## 6. Tradition 5 — Continuous-Time Quantum Walks (CTQW)

### 6.1 Foundational work

**Farhi, E., & Gutmann, S. (1998).** Quantum computation and decision trees.
*Physical Review A* 58(2).

Introduces CTQW. Walker state is complex vector |ψ(t)⟩ evolving under
Hermitian Hamiltonian H:

|ψ(t)⟩ = exp(-iHt) |ψ(0)⟩

For graphs, H = -A (negative adjacency) or H = L (Laplacian) are common
choices.

**Grover, L. K. (1996).** A fast quantum mechanical algorithm for database
search. *STOC 1996.*

Proves quadratic speedup for unstructured search: O(√N) quantum vs O(N)
classical.

**Ambainis, A. (2007).** Quantum Walk Algorithm for Element Distinctness.
*SIAM J. Computing* 37(1).

Provides O(N^(2/3)) quantum algorithm for element distinctness, an improvement
over classical O(N log N).

**Childs, A. M., et al. (2003).** Exponential algorithmic speedup by a quantum
walk. *STOC 2003.*

Proves exponential speedup for specific graph traversal problem (glued-trees).

### 6.2 Classical simulation of quantum walks

**Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review* 45(1).

Surveys numerical methods for matrix exponential. For small matrices
(N ≤ 100), Padé approximation with scaling-and-squaring is most accurate.

**Higham, N. J. (2005, 2008).** Series of papers on numerical matrix
exponential. `scipy.linalg.expm` uses these algorithms.

### 6.3 CTQW applications

**Moore, C., & Russell, A. (2002).** Quantum walks on the hypercube.
*RANDOM 2002.*

Analyzes CTQW on regular graph (hypercube). Proves mixing time faster than
classical.

**Kendon, V. (2007).** Decoherence in quantum walks. *Mathematical Structures
in Computer Science* 17(6).

Studies effect of decoherence on quantum walk advantage.

**Aaronson, S., Chia, N. H., Lin, H. H., Wang, C., & Zhang, R. (2020).** On
the Quantum Complexity of Closest Pair and Related Problems. *CCC.*

Theoretical work on quantum advantage for geometric problems.

### 6.4 CTQW for retrieval

Our search found no prior work applying classical CTQW simulation to:
- Heterogeneous typed knowledge graph retrieval
- Per-query subgraph extraction + CTQW evolution
- Combination with binary hashing for a hybrid retrieval architecture

### 6.5 Gap identified

CTQW research has focused on:
- Regular graph structures (hypercubes, cycles, Johnson graphs)
- Theoretical quantum advantages on quantum hardware
- Decoherence and noise effects

Not found:
- Classical CTQW simulation on irregular heterogeneous knowledge graphs
- Use of CTQW for retrieval refinement (as opposed to global search)
- Empirical results showing MRR or Recall metrics on typed knowledge graphs

---

## 7. Synthesis: the Gap Our Work Fills

The following combination does not appear in the surveyed literature:

| Component | Closest prior | Our use |
|-----------|---------------|---------|
| Binary address from content | SimHash (Charikar 2002) | Same (content-only stage) |
| Per-node graph topology | GCN (Kipf & Welling 2017) | Binary majority vote instead of trained float |
| Associative retrieval | SDM (Kanerva 1988) | Used at the binary node level |
| Compositional binding | XOR binding (Plate 2003) | Applied to retrieval queries |
| Quantum walk retrieval | CTQW (Farhi & Gutmann 1998) | Classical simulation on subgraphs |

The combination of all five in a single retrieval system, with empirical
validation on a heterogeneous typed knowledge graph and multi-architecture
reproducibility, does not appear in any prior work we surveyed.

---

## 8. Threats to Validity of This Review

**Single-author selection.** A single author performed inclusion decisions.
This introduces bias. Mitigations: (a) explicit inclusion/exclusion criteria,
(b) all cited works traceable via DOI, (c) this document public for
correction by community.

**Database limitations.** Google Scholar is comprehensive but not curated.
Semantic Scholar has good coverage of recent work but less of pre-1995 papers.
ACM DL and IEEE Xplore have paywall limitations.

**Time constraint.** The review was conducted in a single session. A more
rigorous PRISMA review would involve second-reviewer verification.

**Language scope.** Only English and Portuguese literature was surveyed. Chinese
and Russian literature on relevant topics (particularly in quantum computing)
was not covered.

---

## 9. Conclusions of the Review

The proposed method combines elements from five prior-art traditions (SDM,
HDC, LSH, GNN, CTQW) in a configuration not previously documented. The
empirical evaluation in the main paper provides the first reported results
on this specific combination.

Further systematic review with multiple reviewers is warranted if the method
is submitted to peer-review venues. This document should be treated as a
structured review, not a PRISMA-compliant systematic review.

---

## Appendix A — Search query log (abbreviated)

```
Query 1: "sparse distributed memory" "graph retrieval"
  Google Scholar results: 127
  Post-screening: 3 (Kanerva 1988, Ramsauer 2020, Poduval 2022)

Query 2: "SimHash" "graph neighbor"
  Google Scholar results: 38
  Post-screening: 2 (Charikar 2002, Manku 2007)

Query 3: "continuous-time quantum walk" "knowledge graph"
  Google Scholar results: 17
  Post-screening: 0 (no direct matches for heterogeneous KG retrieval)

Query 4: "binary embeddings" "graph neural network"
  Google Scholar results: 87
  Post-screening: 3 (Wang 2021, Hamilton 2017, Kipf 2017)

Query 5: "XOR binding" "graph retrieval"
  Google Scholar results: 9
  Post-screening: 2 (Plate 1995, Plate 2003)
```

Full reviewed list of 47 papers is encoded in `references.bib`.

---

*This extended SLR is provided in addition to Section 2 of the main paper
for completeness and reproducibility of the literature review process.*

*DOI: 10.5281/zenodo.19645323*
