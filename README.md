# Tuna

> An optimized FP16 x FP4 gemm kernel.

The configs included here are only tuned for an RTX A6000 (not Ada). If you run the test on anything else you will get less than optimum speeds, and even still, the kernels could use significantly more tuning.

---

# Small Batch \* Seq Benchmarks

All with batch size 16, on RTX A6000, and average of 1000 runs with 100 runs for warmup.

| Benchmark                                     | Tuna Time (ms)      | cuBLAS Time (ms)     | Speedup |
| --------------------------------------------- | ------------------- | -------------------- | ------- |
| Mistral MLP Input (combined gate and up proj) | 0.17720630019903183 | 0.3694557063281536   | 2.08    |
| Mixtral MLP Input (combined all experts)      | 1.086780022829771   | 2.690451130270958    | 2.48    |
| Mistral MLP Output                            | 0.10128284990787506 | 0.17485007271170616  | 1.73    |
| Mixtral MLP Output (combined all experts)     | 0.6302669756114483  | 1.3500527553260326   | 2.14    |
| Mi(s\|x)tral QKV Proj                         | 0.05724561959505081 | 0.08708584681153297  | 1.52    |
| Mi(s\|x)tral Attention Out Proj               | 0.05881141871213913 | 0.053854379802942276 | 0.92    |
| Llama 70b Attn Out Proj                       | 0.11454912275075912 | 0.24153944104909897  | 2.11    |
| Llama 70b QKV Proj                            | 0.12170646712183952 | 0.25466201081871986  | 2.09    |
| Llama 70b MLP Input (combined gate and up)    | 0.6226886250078678  | 1.3868155255913734   | 2.23    |
| LLama 70b MLP Output                          | 0.30147412419319153 | 0.7220751233398914   | 2.40    |
| Mixtral Total GEMMs (32 blocks)               | 58.6593291759491    | 133.8062115907669    | 2.28    |
| Mistral Total GEMMs (32 blocks)               | 12.625478029251099  | 21.92787218093872    | 1.74    |
| Llama 70b Total GEMMs (80 blocks)             | 92.83346712589264   | 208.4073680639267    | 2.24    |

# Future Improvements

- More tuning for more problem sizes and more GPUs
- Better tuning algorithm than evolutionary. Probably a hierarchical grid tuner.
- Optimize the global to shared loading, perhaps with some different cache policies when data isn't reused.
