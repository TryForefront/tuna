/*
 * Copyright (C) 2024 Forefront Industries, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void wrapper(void *q, void *x, void *c, void *s, void *o, int m, int n, int k, cudaStream_t stream);

void matmul(torch::Tensor x, torch::Tensor q, torch::Tensor s, torch::Tensor c, torch::Tensor o, int m, int n, int k)
{

    wrapper(q.data_ptr(),
            x.data_ptr(),
            c.data_ptr(),
            s.data_ptr(),
            o.data_ptr(),
            m,
            n,
            k,
            at::cuda::getCurrentCUDAStream(x.get_device()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul", &matmul, "FP16xFP4 Matrix Multiplication Kernel");
}
