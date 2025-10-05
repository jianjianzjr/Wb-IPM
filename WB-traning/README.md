# Warm-Basis Training (PyTorch)

A PyTorch implementation for training a network that produces warm bases (pilot reconstructions / guidance subspaces) for large-scale linear inverse problems (e.g., Fluorescence Molecular Tomography).

This repo focuses on the learning component that generates warm information consumed later by iterative solvers (e.g., WB-IPM).



# Key feasures

- Angle-based loss that aligns predictions with ground truth (directional supervision), with optional L2 weight decay.

- Simulation-driven dataset (phantom data from STIFT-style generation).

- Clean training/eval scripts with logging and checkpointing.



# Repo structure

```bash
WB-training/
├─ checkpoint/              # saved models and exports (created at runtime)
├─ data/                    # datasets (put/generated here)
├─ log/                     # tensorboard/text logs
├─ data_processing.py       # dataset I/O & transforms
├─ eval.py                  # evaluation (via testing datasets)
├─ model.py                 # baseline network
├─ model_attention.py       # attention-augmented variant
├─ phantomDataProcessing.py # phantom/STIFT data generation helpers
├─ train_simulation.py      # main training script
├─ utils.py                 # misc utilities, seeds, meters, I/O
└─ README.md
```



# Requirements

Python ≥ 3.8

PyTorch ≥ 1.12 (with CUDA if you have a GPU)

torchvision, numpy, scipy, scikit-image, matplotlib, tqdm, tensorboard



# Citation

If this code helps your research, please cite the accompanying paper for the overall warm-basis framework:

Guo, R., Jiang, J., Jin, B., Ren, W., and Zhang, J.
“A Warm-basis Method for Bridging Learning and Iteration: A Case Study in Fluorescence Molecular Tomography,” 2025.



## References

```less
1. Ren, W., Isler, H., Wolf, M., Ripoll, J., & Rudin, M. (2019). "Smart toolkit for fluorescence tomography: simulation, reconstruction, and validation." IEEE Transactions on Biomedical Engineering, 67(1), 16-26.
2. Jin, K. H., McCann, M. T., Froustey, E., Unser, M. "Deep Convolutional Neural Network for Inverse Problems in Imaging." IEEE Trans. Image Processing, 26(9):4509–4522, 2017.
```

