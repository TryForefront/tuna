# Tuna

> An optimized FP16 x FP4 gemm kernel.

The configs included here are only tuned for an RTX A6000 (not Ada). If you run the test on anything else you will get less than optimum speeds, and even still, the kernels could use significantly more tuning.

---

# Small Batch \* Seq Benchmarks

All with batch size 16, on RTX A6000, and average of 1000 runs with 100 runs for warmup.

| Benchmark                                     | Tuna Time (ms)      | cuBLAS Time (ms)     |
| --------------------------------------------- | ------------------- | -------------------- |
| Mistral MLP Input (combined gate and up proj) | 0.17720630019903183 | 0.3694557063281536   |
| Mixtral MLP Input (combined all experts)      | 1.086780022829771   | 2.690451130270958    |
| Mistral MLP Output                            | 0.10128284990787506 | 0.17485007271170616  |
| Mixtral MLP Output (combined all experts)     | 0.6302669756114483  | 1.3500527553260326   |
| QKV Projection                                | 0.05724561959505081 | 0.08708584681153297  |
| Attention Out Projection                      | 0.05881141871213913 | 0.053854379802942276 |
