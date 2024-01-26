import json

out = """#define TRANSFORM_N true
#define TRANSFORM_T false
#define LAYOUT_C true
#define LAYOUT_F false
#define QUANTIZED true
#define NOT_QUANTIZED false

typedef bool MATRIX_QUANTIZE_STATUS;

typedef bool MATRIX_TRANSFORM;

struct KernelConfig
{
    const int tileM;
    const int tileN;
    const int tileK;
    const int patchM;
    const int patchN;
    const int k;
    const int m;
    const int splitK;
    const int warpCountM;
    const int warpCountN;
    const int mmaSizeM;
    const int mmaSizeN;
    const int mmaSizeK;
    const int warpMmaCountM;
    const int warpMmaCountN;
    const int warpMmaCountK;
    const int contiguousBytesA;
    const int contiguousBytesB;
    const int deqBlockSize;

    const int stages;
    const int absMaxPerBlock;
    const int threadsPerBlock;

    const int pipelineStrat;
    const int paddingC;

    static constexpr int codeSize = 16;
    static constexpr bool isSafe = false;
    static constexpr int alignSizeBytesA = 16;
    static constexpr int alignSizeBytesB = 16;

    static constexpr MATRIX_TRANSFORM transformA = TRANSFORM_N;
    static constexpr MATRIX_TRANSFORM transformB = TRANSFORM_T;
    static constexpr MATRIX_TRANSFORM transformC = TRANSFORM_T;

    static constexpr MATRIX_QUANTIZE_STATUS quantStatA = QUANTIZED;
    static constexpr MATRIX_QUANTIZE_STATUS quantStatB = NOT_QUANTIZED;

    static constexpr int deqBlockCount = 1;
};\n\n"""

TEMPLATE = """constexpr KernelConfig $NAME_$n = {
    /* tileM */ $tile_m,
    /* tileN */ $tile_n,
    /* tileK */ $tile_k,
    /* patchM */ $patch_m,
    /* patchN */ $patch_n,
    /* k */ $k,
    /* m */ $m,
    /* splitK */ $split_k,
    /* warpCountM */ $warp_count_m,
    /* warpCountN */ $warp_count_n,
    /* mmaSizeM */ $mma_size_m,
    /* mmaSizeN */ $mma_size_n,
    /* mmaSizeK */ $mma_size_k,
    /* warpMmaCountM */ $warp_mma_count_m,
    /* warpMmaCountN */ $warp_mma_count_n,
    /* warpMmaCountK */ $warp_mma_count_k,
    /* contiguousBytesA */ $contiguous_bytes_A,
    /* contiguousBytesB */ $contiguous_bytes_B,
    /* deqBlockSize */ $deq_block_size,
    /* stages */ $stages,
    /* absMaxPerBlock */ $abs_max_per_block,
    /* threadsPerBlock */ $tpb,
    /* pipelineStrat */ $pipeline_strat,
    /* paddingC */ $padding_C};\n\n"""


with open("param_map.json") as f:
    param_map = json.load(f)


def make_name(m, n, k):
    if m == 4096 and k == 4096:
        return "attn_out"

    if m == 6144 and k == 4096:
        return "qkv_proj"

    if m == 14336 * 2 and k == 4096:
        return "mistral_mlp_in"

    if m == 14336 * 2 * 8 and k == 4096:
        return "mixtral_mlp_in"

    if m == 4096 and k == 14336:
        return "mistral_mlp_out"

    if m == 4096 * 8 and k == 14336:
        return "mixtral_mlp_out"

    raise ValueError(f"Unknown name for m={m}, n={n}, k={k}")


for problem_size, params in param_map.items():
    m, n, k = problem_size.split("_")
    m = int(m)
    n = int(n)
    k = int(k)

    params["m"] = m
    params["n"] = n
    params["k"] = k

    params["NAME"] = make_name(m, n, k)

    this_template = TEMPLATE

    for k, v in params.items():
        this_template = this_template.replace(f"${k}", str(v))

    out += this_template


with open("configs.cu", "w") as f:
    f.write(out)
