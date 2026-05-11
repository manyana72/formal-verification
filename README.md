## Thesis on Formal Verification of Neural Networks

This repository implements both phases of the thesis:

**Phase 1 — ReLU MLP foundations**
- Reproducible environment and a fixed MNIST evaluation split.
- Training of an MNIST MLP + tiny CNN, and PGD \( \ell_\infty \) evaluation on a fixed subset.
- MILP-based exact robustness verification for a ReLU MLP using IBP-tight big-M bounds, with automatic solver backend selection.
- IBP-aware ("certified") training of the MLP so that the IBP-verifiable rate becomes non-trivial at thesis-grade radii.

**Phase 2 — Vision Transformer extension** (principal contribution)
- A ViT-Tiny architecture for MNIST (4×4 patches, \(N=49\) tokens, \(L=2\) blocks, \(H=2\) heads, RMSNorm + ReLU, mean-pool) sized to keep MILP verification tractable.
- Two checkpoints: `vit_standard` (cross-entropy only) and `vit_lipmargin` (Lipschitz-margin loss + spectral-norm regularisation of the attention projections), with Lipschitz constants differing by orders of magnitude.
- A novel **MILP encoding for transformer encoder blocks**. To the best of our knowledge, this is the first MILP-style verifier for self-attention. Curved scalar functions (\(\exp\), \(1/x\), \(1/\sqrt{x}\), \(x^2\)) are sandwiched by piecewise-linear lower and upper envelopes; every bilinear product (\(QK^\top\), \(AV\), the RMSNorm scaling) is wrapped in a McCormick relaxation; all relaxations are tightened by IBP intervals.
- A **compositional hybrid verifier** that runs Lipschitz pre-filter → IBP → PGD → MILP-PWL in increasing cost order and stops at the first stage that decides the sample. Each pairwise cross-method invariant is checked sample-by-sample as a runtime assertion.

### Install

```bash
pip install -e .
```

This expects Python \(\ge 3.10\) with `torch`, `torchvision`, `numpy`, `tqdm`, `pyyaml`, `pandas`, and `ortools` (CBC) available. If `gurobipy` is installed, the verifier can use Gurobi automatically.

### Commands

- **Run pipeline** (ensure split, train MLP quickly if needed, run PGD eval, then MILP verify on a few samples):

```bash
python -m experiments.run_pipeline --config configs/baseline.yaml
```

- **Train MLP only**:

```bash
python -m experiments.train_mnist --config configs/mlp.yaml
```

- **PGD evaluation** on fixed subset:

```bash
python -m experiments.attack_pgd --ckpt runs/.../model.pt --eps 0.03 --subset assets/splits/mnist_eval_100.json
```

- **MILP verification** of the ReLU MLP:

```bash
python -m experiments.verify_milp --ckpt runs/.../model.pt --eps 0.03 --solver auto --time-limit 30 --max-samples 5
```

### Phase 2 — Vision Transformer pipeline

Phase 2 is delivered through self-contained Colab notebooks in `notebooks/`. Each notebook embeds the library code it needs inline, so they can be run top-to-bottom without a global `pip install -e .`.

| Notebook | Purpose |
| --- | --- |
| `09_p2_vit_train_standard.ipynb`  | Train `vit_standard` with cross-entropy only. |
| `10_p2_vit_train_lipmargin.ipynb` | Train `vit_lipmargin` with a Lipschitz-margin loss + spectral-norm regularisation. |
| `11_p2_lipschitz_bounds.ipynb`    | Compute \(L_{\text{total}}\) for both checkpoints and run the Lipschitz pre-filter \( m_y(x_0) > L_{\text{total}}\,\varepsilon\sqrt{D} \Rightarrow \) **VERIFIED**. |
| `12_p2_ibp_vit.ipynb`             | IBP over the full ViT (patch embedding → RMSNorm → attention → MLP → mean-pool → logits). |
| `13_p2_pgd_vit.ipynb`             | PGD \(\ell_\infty\) attack on the ViT: 100 steps, \(\alpha = 0.1\varepsilon\), 5 random restarts. |
| `14_p2_milp_pwl_primitives.ipynb` | Piecewise-linear lower/upper envelopes for \(\exp\), \(1/x\), \(1/\sqrt{x}\), \(x^2\) on IBP-derived ranges, with per-call PWL gap reporting. |
| `15_p2_milp_rmsnorm.ipynb`        | MILP encoding of RMSNorm via the PWL primitives and McCormick scaling. |
| `16_p2_milp_attention.ipynb`      | MILP encoding of a single self-attention head: \(Q,K,V\) projections (linear), score products \(QK^\top\) (McCormick), shift + softmax (PWL on \(\exp\), linear sum, PWL on \(1/x\), McCormick attention weight), output products \(AV\) (McCormick, tightened by \(A\in[0,1]\)), and concat + output projection (linear). |
| `17_p2_milp_vit_full.ipynb`       | Full ViT-Tiny MILP encoder, per-class margin objective \(\max(z_c - z_y)\), verdict ∈ {VERIFIED, FALSIFIED, INCONCLUSIVE, TIMEOUT}. |
| `18_p2_hybrid_verifier.ipynb`     | Compositional pipeline Lipschitz → IBP → PGD → MILP-PWL, with the **(I1)–(I3)** invariants asserted at runtime. |
| `19_p2_visualise_results.ipynb`   | Figures for the thesis report and poster: hybrid-stage breakdown, robustness vs. \(\varepsilon\), per-call PWL gaps, timing breakdown. |

The Phase 2 design notes (architectural choices, soundness arguments, encoding details) are in `configs/PLAN2_transformer_extension.md`.

### Notes

- The fixed MNIST evaluation subset is stored at `assets/splits/mnist_eval_100.json` with format `{"seed": 1234, "indices": [...100 ints...]}`. Code will regenerate it if missing, but the default file is committed for full reproducibility.
- The Phase 1 MILP verifier encodes the MLP with big-M ReLU constraints using pre-activation bounds from interval bound propagation (IBP). For each class \(c \ne y\), it maximizes the margin \( \text{logit}_c - \text{logit}_y \); if the worst margin is \(\le 0\), the example is **VERIFIED**, otherwise a falsifying adversarial example is extracted and validated through the actual network.
- The Phase 2 MILP verifier extends this idea to a ViT encoder block. Each PWL envelope sandwiches the true curve from below and above; each McCormick wraps a bilinear product in four linear inequalities. Because every relaxation is an outer relaxation of the true forward pass, a non-positive optimum on every wrong class still soundly certifies robustness (up to the explicit PWL gap, which is reported alongside every verdict).
- Cross-method invariants checked at runtime: **(I1)** Lipschitz-VER ⇒ IBP-VER and MILP-VER; **(I2)** IBP-VER ⇒ MILP-VER (up to the PWL gap); **(I3)** PGD-FAL ⇒ MILP-FAL or TIMEOUT. A single violation would falsify the encoding.
- Solver backend selection is automatic via `--solver auto`: it uses Gurobi if `gurobipy` is available, otherwise OR-Tools CBC. A paid Gurobi licence is recommended for Phase 2, where model sizes exceed the free CBC's practical scaling.

