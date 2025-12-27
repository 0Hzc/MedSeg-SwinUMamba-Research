# Experimental Pipeline
## 1. Environment
- Dependencies: `causal-conv1d` and `mamba-ssm` must match CUDA.
- Dataset: BraTS 2024 or ACDC.
## 2. Memory Optimization (A40)
- Always use `torch.cuda.amp.autocast`.
- If OOM, increase `num_batches_per_epoch` instead of batch size.
