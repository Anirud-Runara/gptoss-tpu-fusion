"""
Build the V3 denominator/RMSNorm CUDA extension (`denominator_cuda`).

    GPTOSS_CUDA_ARCH=sm_90 pip install -e .      # Hopper (H100)
    GPTOSS_CUDA_ARCH=sm_80 pip install -e .      # Ampere (A100)
    pip install -e .                             # let torch pick the arch

You usually don't need this — backends/cuda/load_cuda.py JIT-builds the
extension on first import. Use this for an explicit ahead-of-time build.
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_nvcc = ["-O3"]
_arch = os.environ.get("GPTOSS_CUDA_ARCH")
if _arch:
    _nvcc.append(f"-arch={_arch}")

setup(
    name="denominator_cuda",
    ext_modules=[
        CUDAExtension(
            name="denominator_cuda",
            sources=["csrc/denominator.cpp", "csrc/denominator_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": _nvcc},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
