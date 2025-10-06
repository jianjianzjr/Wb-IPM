# Warm-Basis Iterative Projection Method

**Goal:** Bridge learning and iteration for large-scale linear inverse problems  
(case study: Fluorescence Molecular Tomography, FMT).

This repo has two parts:

- **`Wb-training/`** — PyTorch code to train a network that produces **warm bases**  
  (pilot reconstructions / subspace hints).
- **`Wb-IPM/`** — MATLAB implementation of the **Warm-Basis Iterative Projection Method**  
  (hybrid projection with flexible Golub–Kahan).

---

## Repository Structure
```text
.
├─ Wb-IPM/          # MATLAB solver (demos, core routines)
├─ Wb-training/     # PyTorch training (models, scripts, data utils)
└─ README.md
```

## Requirements

Wb-training (Python): Python ≥ 3.8, PyTorch ≥ 1.12

Wb-IPM (MATLAB): R2022b+

## Citation
```text
Guo, R., Jiang, J., Jin, B., Ren, W., Zhang, J.
“A Warm-basis Method for Bridging Learning and Iteration: A Case Study in Fluorescence Molecular Tomography,” 2025.
```
