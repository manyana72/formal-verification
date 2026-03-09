## Thesis on Formal Verification of Neural Networks

This repository implements the following:

- Create a reproducible environment and fixed MNIST evaluation split.
- Training MNIST MLP + tiny CNN, and PGD \( \ell_\infty \) evaluation on a fixed subset.
- MILP-based robustness verification for a ReLU MLP using IBP-tight big‑M bounds, with automatic solver backend selection.

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

### Notes

- The fixed MNIST evaluation subset is stored at `assets/splits/mnist_eval_100.json` with format `{"seed": 1234, "indices": [...100 ints...]}`. Code will regenerate it if missing, but the default file is committed for full reproducibility.
- The MILP verifier encodes the MLP with big‑M ReLU constraints using pre-activation bounds from interval bound propagation (IBP). For each class \(c \ne y\), it maximizes the margin \( \text{logit}_c - \text{logit}_y \); if the worst margin is \(\le 0\), the example is **VERIFIED**, otherwise a falsifying adversarial example is extracted.
- Solver backend selection is automatic via `--solver auto`: it uses Gurobi if `gurobipy` is available, otherwise OR-Tools CBC.

