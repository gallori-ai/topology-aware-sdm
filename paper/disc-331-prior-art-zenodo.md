# H6: Thermal-Substrate Computing
## Harnessing Environment-Assisted Quantum Transport and Collective Biological Thermoregulation for Energy-Efficient Chip Architecture

**Author:** Cleber Barcelos Costa  
**Affiliation:** Gallori AI  
**ORCID:** 0009-0000-5172-9019  
**Date of First Registration:** 2026-04-17 (internal knowledge graph, Gallori AI)  
**Date of Public Deposit:** 2026-04-28  
**License:** CC BY 4.0  
**Upload Type:** Technical Note / Preprint  
**Keywords:** ENAQT, thermal substrate computing, quantum biology, energy-efficient computing, chip design, collective thermoregulation, Landauer limit, bisociation, H6, quantum walk, thermal gradient

---

## Abstract

This technical note registers the original hypothesis H6 — Thermal-Substrate Computing —
arising from a bisociation between two independently studied biological mechanisms and
the domain of energy-efficient chip design.

**The bisociation:**

Domain A: Biological thermoregulation via Environment-Assisted Quantum Transport (ENAQT).
Photosynthesis achieves ~95% energy transfer efficiency at 37°C precisely because thermal
noise — rather than being suppressed — acts as a transport resource. The Fenna-Matthews-Olson
(FMO) complex exploits quantum coherence modulated by thermal fluctuations. Additionally,
collective animal thermoregulation (e.g., bird flocks, insect colonies) operates without
a central controller: load distributes by local thermal state, producing globally stable
temperatures via purely local interactions.

Domain B: Energy-efficient computing. Modern chips treat thermal gradients as waste to be
managed and removed. Active cooling consumes 30–40% of data center energy. The Landauer
limit (kT ln2 per irreversible bit operation ≈ 2.75 × 10⁻²¹ J at 300 K) defines the
theoretical minimum; current architectures are 4–5 orders of magnitude above this limit,
largely because thermal energy is dissipated twice: once during computation, once during
cooling.

**The hypothesis:**

The same mechanism that makes ENAQT efficient — thermal noise as transport resource, not
adversary — and the same load-distribution principle that governs collective thermoregulation —
local thermal state as the routing signal, no central orchestration — can be applied
structurally to chip design.

Formally:

```
ENAQT_thermal_noise_as_transport
    :: chip_thermal_gradient_as_computational_substrate

collective_thermoregulation_no_central_controller
    :: distributed_load_balancing_by_local_thermal_state

biological_fever (accelerated metabolism at elevated temperature)
    :: accelerated_computation_in_thermal_substrate_mode

UTXO_append_only (energy state register, non-destructive)
    :: thermal_state_as_computational_register
```

In this architecture, thermal gradient is not an obstacle to computation — it is the
computational substrate. Higher local temperature in a specific region means more available
computational resource for thermally-adapted processes, not system failure or throttling.

**The prediction:**

Chips architectured under the Thermal-Substrate Computing model approach the Landauer
efficiency limit (kT ln2 per bit) without requiring active cooling in the designed
operating range. Energy is dissipated once (as useful computation), not twice (computation
+ cooling). Efficiency increases with temperature within the designed operating range —
the opposite of conventional architecture behavior.

**The falsifiable test (H6-T1):**

Simulate an ENAQT-inspired circuit in thermal-substrate mode. Measure energy dissipation
versus a conventional equivalent circuit at T = 25°C, 50°C, and 75°C.

Falsification criterion:
- If `energy_efficiency(ENAQT_thermal_substrate) > energy_efficiency(conventional)` at T > 50°C:
  H6 supported.
- If efficiency decreases or equals conventional at all temperatures:
  H6 falsified. New hypothesis required.

**DVU estimate (internal):** 9.8 — novelty: critical, breadth: critical, evidence: medium,
falsifiability: clear, actionability: high.

**Scope of this registration:**

This document registers the hypothesis and the formal structural mapping for prior art
purposes. It does not claim a reduction to practice, a working implementation, or any
specific circuit design. The novelty claim is the structural bisociation itself: applying
the ENAQT + collective thermoregulation mechanism to chip architecture as a unified
computational paradigm.

---

## Prior Work (what this is NOT)

- **ENAQT / FMO complex** (Engel et al. 2007, Fleming group): established quantum biology
  result. Not claimed here. Used as source domain.
- **Thermal management in computing** (general): extensive literature. Not claimed here.
- **Collective thermoregulation in biology**: known biological phenomenon. Not claimed here.
- **Landauer limit**: established thermodynamics (Landauer 1961). Not claimed here.

**The novel contribution** is exclusively the structural mapping — the application of the
ENAQT mechanism and the collective thermoregulation load-distribution principle as a
unified architectural paradigm for chip design, and the specific prediction that this
yields efficiency gains with temperature rather than efficiency losses.

---

## Chain Context (Gallori AI internal reference)

This hypothesis is H6 in the quantum biology chain:

```
DISC-032 → DISC-068 (H1 ENAQT) → DISC-071 (H2 CQW)
    → DISC-106 (H3 algorithmic inversion) → DISC-107 (H4 density-phase)
    → DISC-138 (H5 verified prior art) → DISC-331 (H6 — this document)
```

H1 (ENAQT as algorithmic transport) was internally falsified as a standalone claim and
superseded by H2–H5, which provided the validated chain from which H6 emerges.

---

## Declaration

I, Cleber Barcelos Costa, declare that this hypothesis was generated independently within
the Gallori AI knowledge discovery engine (CEI — Cognitive Enterprise Intelligence) on
2026-04-17, prior to any public disclosure. This Zenodo deposit constitutes the first
public registration for prior art purposes.

IP ownership: Gallori AI  
IP model: Model A (internal discovery, 100% Gallori AI)  
Jurisdiction: Brazil, United States, European Union  

---

*Gallori AI — Betim, Minas Gerais, Brazil*  
*First registered: 2026-04-17 | First public deposit: 2026-04-28*
