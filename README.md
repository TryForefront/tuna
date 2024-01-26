# Tuna

> An optimized FP16 x FP4 gemm kernel.

The configs included here are only tuned for an RTX A6000 (not Ada). If you run the test on anything else you will get less than optimum speeds, and even still, the kernels could use significantly more tuning.

---

# Small Batch \* Seq Benchmarks

All with batch size 16, on RTX A6000, and average of 1000 runs with 100 runs for warmup.

### Mistral MLP Input (combined gate and up proj)

- Tuna Time: 0.16865141689777374 ms
- cuBLAS Time: 0.3705862909555435 ms

### Mixtral MLP Input (combined all experts)

- Tuna Time: 1.5256261825561523 ms
- cuBLAS Time: 2.692037969827652 ms

### Mistral MLP Output

- Tuna Time: 0.10315269231796265 ms
- cuBLAS Time: 0.1758352667093277 ms

### Mixtral MLP Output (combined all experts)

- Tuna Time: 0.6196577101945877 ms
- cuBLAS Time: 1.3504721596837044 ms

### QKV Projection

- Tuna Time: 0.06974749267101288 ms
- cuBLAS Time: 0.08826814591884613 ms

### Attention Out Projection

- Tuna Time: 0.10606057941913605 ms
- cuBLAS Time: 0.05419932305812836 ms
