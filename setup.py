from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

torch.utils.cpp_extension.COMMON_NVCC_FLAGS = [
    "--expt-relaxed-constexpr"
]  # This is necessary to enable half precision conversions

setup(
    name="tuna",
    ext_modules=[
        CUDAExtension(
            "tuna",
            ["extension.cpp", "kernel.cu", "configs.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
