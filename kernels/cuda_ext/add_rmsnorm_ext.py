"""CUDA C++ fused add plus RMSNorm, built on first import with torch.utils.cpp_extension.load.

Same signature and autograd behavior as kernels.triton_ops.add_rmsnorm:
    y, s = add_rmsnorm(a, r, w, eps=1e-5, out_dtype=None)

Needs nvcc (CUDA toolkit 12.8 or newer for sm_120) and ninja. The compiled module is
cached under ~/.cache/torch_extensions, so only the first import pays the build time.
"""

import os
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().with_name("add_rmsnorm.cu")


def _find_cuda_home():
    """Use CUDA_HOME if set, otherwise the newest /usr/local/cuda-12.x install."""
    if os.environ.get("CUDA_HOME"):
        return
    cands = sorted(Path("/usr/local").glob("cuda-1[2-9].*")) or [Path("/usr/local/cuda")]
    for c in reversed(cands):
        if (c / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(c)
            os.environ["PATH"] = f"{c / 'bin'}:{os.environ.get('PATH', '')}"
            return


def _build():
    _find_cuda_home()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"   # build only for this GPU
    from torch.utils.cpp_extension import load   # imported after CUDA_HOME is set, it reads it at import
    return load(
        name="add_rmsnorm_cuda",
        sources=[str(_SRC)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
        verbose=os.environ.get("VERBOSE_BUILD") == "1",
    )


_ext = _build()


class _AddRMSNormCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, r, w, eps, out_dtype):
        shape, D = r.shape, r.shape[-1]
        a2 = a.reshape(-1, D).contiguous()
        r2 = r.reshape(-1, D).contiguous()
        y_dtype = out_dtype or a.dtype
        y, s, rstd = _ext.forward(a2, r2, w.float().contiguous(), float(eps), y_dtype == torch.bfloat16)
        ctx.save_for_backward(s, w, rstd)
        ctx.dtypes = (a.dtype, r.dtype)
        return y.view(*shape[:-1], D), s.view(*shape[:-1], D)

    @staticmethod
    def backward(ctx, dy, ds_in):
        s, w, rstd = ctx.saved_tensors
        D = s.shape[-1]
        if dy is None:
            dy = torch.zeros_like(ds_in)
        ds2 = ds_in.reshape(-1, D).contiguous().to(s.dtype) if ds_in is not None else None
        a_dtype, r_dtype = ctx.dtypes
        da, dr, dwp = _ext.backward(dy.reshape(-1, D).contiguous(), ds2, s, w.float().contiguous(), rstd,
                                    a_dtype == torch.bfloat16, r_dtype == torch.bfloat16)
        return da.view(dy.shape), dr.view(dy.shape), dwp.sum(0).to(w.dtype), None, None


def add_rmsnorm(a, r, w, eps=1e-5, out_dtype=None):
    """s = a + r, y = rmsnorm(s) * w. Returns (y, s). Same contract as the Triton version."""
    return _AddRMSNormCUDA.apply(a, r, w, eps, out_dtype)
