# Tuna

> An optimized FP16 x FP4 gemm kernel.

The configs included here are only tuned for an RTX A6000 (not Ada). If you run the test on anything else you will get less than optimum speeds, and even still, the kernels could use significantly more tuning.

---

# Small Batch \* Seq Benchmarks

All with batch size 16, on RTX A6000, and average of 1000 runs with 100 runs for warmup. These configs are what are currently found in the repo's `param_map.json` and `configs.cu`

| Benchmark                                     | Tuna Time (ms)       | cuBLAS Time (ms)    | Speedup |
| --------------------------------------------- | -------------------- | ------------------- | ------- |
| Mistral MLP Input (combined gate and up proj) | 0.16256877034902573  | 0.3670984320342541  | 2.26    |
| Mixtral MLP Input (combined all experts)      | 1.0869103781878948   | 2.6869438253343105  | 2.47    |
| Mistral MLP Output                            | 0.10303235054016113  | 0.17698441073298454 | 1.72    |
| Mixtral MLP Output (combined all experts)     | 0.6412508971989155   | 1.3513591587543488  | 2.11    |
| Mi(s\|x)tral QKV Proj                         | 0.05689075216650963  | 0.08756674453616142 | 1.54    |
| Mi(s\|x)tral Attention Out Proj               | 0.048066891729831696 | 0.05451885610818863 | 1.13    |
| Llama 70b Attn Out Proj                       | 0.1169043518602848   | 0.24606598913669583 | 2.10    |
| Llama 70b QKV Proj                            | 0.1213807724416256   | 0.25460585579276085 | 2.10    |
| Llama 70b MLP Input (combined gate and up)    | 0.5466882549226284   | 1.3823552504181862  | 2.53    |
| LLama 70b MLP Output                          | 0.3022856377065182   | 0.7248534262180328  | 2.40    |
| Mixtral Total GEMMs (32 blocks)               | 58.65980541706085    | 134.82996714115143  | 2.30    |
| Mistral Total GEMMs (32 blocks)               | 11.857880473136902   | 23.014922618865967  | 1.94    |
| Llama 70b Total GEMMs (80 blocks)             | 86.98072135448456    | 208.63044117253     | 2.39    |

# Future Improvements

- More tuning for more problem sizes and more GPUs
- Better tuning algorithm than evolutionary. Probably a hierarchical grid tuner.
- Optimize the global to shared loading, perhaps with some different cache policies when data isn't reused.
