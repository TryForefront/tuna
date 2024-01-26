#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void wrapper(void *q, void *x, void *c, void *s, void *o, int m, int n, int k);

void matmul(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor c, torch::Tensor o, int m, int n, int k)
{

    wrapper(q.data_ptr(),
            x.data_ptr(),
            c.data_ptr(),
            s.data_ptr(),
            o.data_ptr(),
            m,
            n,
            k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul", &matmul, "FP16xFP4 Matrix Multiplication Kernel");
}
