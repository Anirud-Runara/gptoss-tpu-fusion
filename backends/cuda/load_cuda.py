"""
Load the V3 denominator/RMSNorm CUDA extension, JIT-building if needed.

Ported from the Qwen3 repo. Two changes for this repo:
  * _ROOT resolves to the repo root (csrc/ lives there) from backends/cuda/.
  * The SM arch is configurable via the GPTOSS_CUDA_ARCH env var instead of a
    hardcoded sm_120, so it builds on whatever GPU you run. If unset, we let
    torch pick (TORCH_CUDA_ARCH_LIST / auto). Common values:
        sm_100 / sm_100a  -> Blackwell datacenter (B100 / B200 / GB200)
        sm_120            -> Blackwell consumer (RTX 50xx) [the Qwen3 default]
        sm_90             -> Hopper (H100)
        sm_80             -> Ampere (A100)
    The kernel is generic CUDA (Welford + warp shuffles), so it is portable
    across these archs. Blackwell requires CUDA toolkit >= 12.8 and a cu128+
    torch build.

    GPTOSS_CUDA_ARCH=sm_100 python ...   # force Blackwell datacenter
"""
import os
import torch.utils.cpp_extension as ext

# Bypass the strict CUDA toolkit/driver version check (forward-compatible drivers).
_orig_check = ext._check_cuda_version
ext._check_cuda_version = lambda *a, **k: None

# backends/cuda/load_cuda.py -> backends/cuda -> backends -> repo root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_cuda_flags = ["-O3"]
_arch = os.environ.get("GPTOSS_CUDA_ARCH")
if _arch:
    _cuda_flags.append(f"-arch={_arch}")

denominator_cuda = ext.load(
    name="denominator_cuda",
    sources=[
        os.path.join(_ROOT, "csrc", "denominator.cpp"),
        os.path.join(_ROOT, "csrc", "denominator_kernel.cu"),
    ],
    extra_cuda_cflags=_cuda_flags,
    extra_cflags=["-O3"],
    verbose=False,
)

ext._check_cuda_version = _orig_check
