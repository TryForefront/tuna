import torch
import tuna
import math
import time


def quantize_blockwise_dynamic_4bit(
    inp: torch.tensor, block_size: int = 4096, preprocess=False
):
    assert inp.numel() % 128 == 0
    assert block_size % 4 == 0
    device = inp.device
    n = inp.numel() // block_size
    out_shape = inp.shape[:-1] + (inp.shape[-1],)

    temp = inp.clone().flatten().reshape(n, -1).to(device)

    absmax = torch.max(temp.abs(), dim=1)[0].float()
    code = torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        device="cuda",
        dtype=torch.float,
    )

    normalized = torch.divide(temp, absmax.unsqueeze(-1)).flatten()
    chunk_size = 512 * 1024
    chunks = math.ceil(normalized.numel() / chunk_size)
    results = torch.empty(size=(normalized.numel(),), dtype=torch.uint8, device=device)

    for i in range(chunks):
        begin = i * chunk_size
        end = min((i + 1) * chunk_size, normalized.numel())
        candidates = torch.sub(
            normalized[begin:end].unsqueeze(0), code.unsqueeze(-1)
        ).abs()
        results[begin:end] = torch.argmin(candidates, keepdim=True, dim=0).flatten()

    if preprocess:
        m, k = inp.shape[-2:]

        results = (
            results.reshape(-1, m // 8, 8, k // 8, 4, 2)
            #        [0,  1,   2,  3,    4, 5]
            .permute(0, 1, 3, 2, 4, 5)  # [-1, m//8, k//8, 8, 4, 2]
            .reshape(-1, m // 8, k // 8, 32, 2)
        )
        results = results[..., 0] * 16 + results[..., 1]  # [-1, m//8, k//8, 32]
        results = results.reshape(*inp.shape[:-2], m // 8, k // 8, 32)
    # else:
    #   results = results.reshape(*out_shape)
    return results, (absmax, code)


if __name__ == "__main__":
    print(
        "if you are running this on anything other than an A6000, you will get less than optimum speeds. The configs have only been tuned for A6000"
    )
    M, N, K = 14336 * 2 * 8, 16, 4096
    x = torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.1
    w = torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.05

    out = torch.zeros((N, M), dtype=torch.float16, device="cuda")

    q, (a, c) = quantize_blockwise_dynamic_4bit(w, block_size=K, preprocess=True)

    baseline = torch.matmul(x, w.t())

    tuna.matmul(x, q, a, c, out, M, N, K)

    print(baseline)
    print(out)

    N_RUNS = 100

    torch.cuda.synchronize()

    # warmup

    for i in range(100):
        out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
        tuna.matmul(x, q, a, c, out, M, N, K)

    torch.cuda.synchronize()

    start = time.perf_counter()

    for i in range(N_RUNS):
        out = torch.zeros((N, M), dtype=torch.float16, device="cuda")
        tuna.matmul(x, q, a, c, out, M, N, K)

    torch.cuda.synchronize()

    end = time.perf_counter()

    print(f"Tuna Time: {(end - start) / N_RUNS * 1000} ms")

    torch.cuda.synchronize()

    start = time.perf_counter()

    for i in range(N_RUNS):
        out = torch.matmul(x, w.t())

    torch.cuda.synchronize()

    end = time.perf_counter()

    print(f"Baseline Time: {(end - start) / N_RUNS * 1000} ms")
