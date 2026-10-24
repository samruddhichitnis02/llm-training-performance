// Fused residual add plus RMSNorm, forward and backward, in CUDA C++.
//
//   s = a + r,  rstd = 1 / sqrt(mean(s^2) + eps),  y = s * rstd * w
//
// Forward: one block per row. Each thread owns VPT groups of 4 columns, loads them with
// one vectorized load per group (16 bytes for fp32, 8 bytes for bf16), keeps s in
// registers, and the sum of squares is reduced with warp shuffles plus a small shared
// memory array for the cross warp step.
//
// Backward: persistent grid of about 4 x SM count blocks. Each block loops over rows
// pid, pid + P, ... With g = dy * w and c = mean(g * s):
//   ds_total = rstd * g - s * rstd^3 * c + ds_in      (same value goes to a and r)
// Every thread owns the same columns in every row, so the weight gradient
// dw = sum over rows of dy * s * rstd accumulates in registers and is written once
// per block into a (P, D) buffer that the host sums.
//
// All math is float32. Loads and stores use each tensor's own dtype (fp32 or bf16),
// so a bf16 sublayer output can be added to an fp32 residual stream directly.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float to_f(float x) { return x; }
__device__ __forceinline__ float to_f(bf16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ bf16 from_f<bf16>(float x) { return __float2bfloat16(x); }

// Four values packed so a single aligned load or store moves all of them.
template <typename T> struct alignas(sizeof(T) * 4) Pack4 { T v[4]; };

template <typename T> __device__ __forceinline__ void load4(const T* p, float out[4]) {
  Pack4<T> pk = *reinterpret_cast<const Pack4<T>*>(p);
#pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = to_f(pk.v[i]);
}

template <typename T> __device__ __forceinline__ void store4(T* p, const float in[4]) {
  Pack4<T> pk;
#pragma unroll
  for (int i = 0; i < 4; ++i) pk.v[i] = from_f<T>(in[i]);
  *reinterpret_cast<Pack4<T>*>(p) = pk;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Sum over the whole block, result returned to every thread.
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  __syncthreads();                      // previous call may still be reading partial[0]
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < nwarps) ? partial[lane] : 0.f;
    v = warp_sum(v);
    if (lane == 0) partial[0] = v;
  }
  __syncthreads();
  return partial[0];
}

template <typename TA, typename TR, typename TS, typename TY, int VPT>
__global__ void __launch_bounds__(512) add_rmsnorm_fwd_kernel(const TA* __restrict__ a, const TR* __restrict__ r,
                                       const float* __restrict__ w, TY* __restrict__ y,
                                       TS* __restrict__ s, float* __restrict__ rstd_out,
                                       int D, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * D;
  float sv[VPT][4];
  float sumsq = 0.f;
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int c = (threadIdx.x + k * blockDim.x) * 4;
    if (c < D) {
      float av[4], rv[4];
      load4(a + base + c, av);
      load4(r + base + c, rv);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sv[k][i] = av[i] + rv[i];
        sumsq += sv[k][i] * sv[k][i];
      }
      store4(s + base + c, sv[k]);
    }
  }
  const float rstd = rsqrtf(block_sum(sumsq) / D + eps);
  if (threadIdx.x == 0) rstd_out[blockIdx.x] = rstd;
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int c = (threadIdx.x + k * blockDim.x) * 4;
    if (c < D) {
      float wv[4], out[4];
      load4(w + c, wv);
#pragma unroll
      for (int i = 0; i < 4; ++i) out[i] = sv[k][i] * rstd * wv[i];
      store4(y + base + c, out);
    }
  }
}

template <typename TA, typename TR, typename TS, typename TY, int VPT>
__global__ void __launch_bounds__(512) add_rmsnorm_bwd_kernel(const TY* __restrict__ dy, const TS* __restrict__ ds_in,
                                       const TS* __restrict__ s, const float* __restrict__ w,
                                       const float* __restrict__ rstd, TA* __restrict__ da,
                                       TR* __restrict__ dr, float* __restrict__ dw_partial,
                                       int64_t N, int D, bool has_ds) {
  float wv[VPT][4], dw[VPT][4];
#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int c = (threadIdx.x + k * blockDim.x) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) dw[k][i] = 0.f;
    if (c < D) load4(w + c, wv[k]);
  }

  // The loop bound is the same for every thread in the block, so block_sum is safe inside it.
  for (int64_t row = blockIdx.x; row < N; row += gridDim.x) {
    const int64_t base = row * D;
    const float rs = rstd[row];
    float g[VPT][4], sv[VPT][4];
    float dot = 0.f;
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int c = (threadIdx.x + k * blockDim.x) * 4;
      if (c < D) {
        float dyv[4];
        load4(dy + base + c, dyv);
        load4(s + base + c, sv[k]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          g[k][i] = dyv[i] * wv[k][i];
          dot += g[k][i] * sv[k][i];
          dw[k][i] += dyv[i] * sv[k][i] * rs;
        }
      }
    }
    const float cmean = block_sum(dot) / D;
    const float rs3 = rs * rs * rs;
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
      const int c = (threadIdx.x + k * blockDim.x) * 4;
      if (c < D) {
        float ds[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) ds[i] = rs * g[k][i] - sv[k][i] * rs3 * cmean;
        if (has_ds) {
          float dsv[4];
          load4(ds_in + base + c, dsv);
#pragma unroll
          for (int i = 0; i < 4; ++i) ds[i] += dsv[i];
        }
        store4(dr + base + c, ds);
        store4(da + base + c, ds);
      }
    }
  }

#pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int c = (threadIdx.x + k * blockDim.x) * 4;
    if (c < D) store4(dw_partial + static_cast<int64_t>(blockIdx.x) * D + c, dw[k]);
  }
}

// Threads per block and 4-wide groups per thread for a row of width D.
// Capped at 512 threads so the backward (about 16 live floats per group) keeps its
// registers: 1024 threads would allow only 64 registers each and fail to launch.
constexpr int kMaxThreads = 512;

void launch_shape(int D, int& threads, int& vpt) {
  const int groups = D / 4;
  vpt = 1;
  while (groups > kMaxThreads * vpt) vpt *= 2;
  threads = (groups + vpt - 1) / vpt;
  threads = std::max(32, ((threads + 31) / 32) * 32);
}

template <typename T> T* ptr(const torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

c10::ScalarType st(bool is_bf16) { return is_bf16 ? torch::kBFloat16 : torch::kFloat32; }

// Calls f with C++ types for (a, r, y). s is the wider of a and r.
template <typename F> void dispatch(bool a_bf16, bool r_bf16, bool y_bf16, F f) {
  auto with_y = [&](auto ta, auto tr) {
    if (y_bf16) f(ta, tr, bf16{}); else f(ta, tr, float{});
  };
  auto with_r = [&](auto ta) {
    if (r_bf16) with_y(ta, bf16{}); else with_y(ta, float{});
  };
  if (a_bf16) with_r(bf16{}); else with_r(float{});
}

template <typename TA, typename TR>
using wider_t = std::conditional_t<std::is_same_v<TA, float> || std::is_same_v<TR, float>, float, bf16>;

void check_input(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == torch::kFloat32 || t.scalar_type() == torch::kBFloat16,
              name, " must be float32 or bfloat16");
}

}  // namespace

// a, r: (N, D). w: (D,) float32. Returns {y, s, rstd}.
std::vector<torch::Tensor> forward(torch::Tensor a, torch::Tensor r, torch::Tensor w, double eps, bool y_bf16) {
  check_input(a, "a");
  check_input(r, "r");
  TORCH_CHECK(a.dim() == 2 && a.sizes() == r.sizes(), "a and r must be (N, D) with the same shape");
  TORCH_CHECK(w.scalar_type() == torch::kFloat32 && w.is_contiguous(), "w must be contiguous float32");
  const int64_t N = a.size(0);
  const int D = static_cast<int>(a.size(1));
  TORCH_CHECK(D % 4 == 0, "D must be a multiple of 4 for vectorized loads");
  TORCH_CHECK(D <= kMaxThreads * 8 * 4, "D up to 16384 supported");
  const c10::cuda::CUDAGuard guard(a.device());

  const bool a_bf16 = a.scalar_type() == torch::kBFloat16;
  const bool r_bf16 = r.scalar_type() == torch::kBFloat16;
  const bool s_bf16 = a_bf16 && r_bf16;
  auto opts = a.options();
  auto y = torch::empty({N, D}, opts.dtype(st(y_bf16)));
  auto s = torch::empty({N, D}, opts.dtype(st(s_bf16)));
  auto rstd = torch::empty({N}, opts.dtype(torch::kFloat32));
  if (N == 0) return {y, s, rstd};

  int threads, vpt;
  launch_shape(D, threads, vpt);
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch(a_bf16, r_bf16, y_bf16, [&](auto ta, auto tr, auto ty) {
    using TA = decltype(ta);
    using TR = decltype(tr);
    using TY = decltype(ty);
    using TS = wider_t<TA, TR>;
    auto args = std::make_tuple(ptr<TA>(a), ptr<TR>(r), ptr<float>(w), ptr<TY>(y), ptr<TS>(s),
                                ptr<float>(rstd), D, static_cast<float>(eps));
    auto go = [&](auto kernel) {
      std::apply([&](auto... xs) { kernel<<<N, threads, 0, stream>>>(xs...); }, args);
    };
    if (vpt == 1) go(add_rmsnorm_fwd_kernel<TA, TR, TS, TY, 1>);
    else if (vpt == 2) go(add_rmsnorm_fwd_kernel<TA, TR, TS, TY, 2>);
    else if (vpt == 4) go(add_rmsnorm_fwd_kernel<TA, TR, TS, TY, 4>);
    else go(add_rmsnorm_fwd_kernel<TA, TR, TS, TY, 8>);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, s, rstd};
}

// dy: (N, D) in y's dtype. ds_in: optional (N, D) in s's dtype. Returns {da, dr, dw_partial}.
std::vector<torch::Tensor> backward(torch::Tensor dy, c10::optional<torch::Tensor> ds_in, torch::Tensor s,
                                    torch::Tensor w, torch::Tensor rstd, bool a_bf16, bool r_bf16) {
  check_input(dy, "dy");
  check_input(s, "s");
  const int64_t N = s.size(0);
  const int D = static_cast<int>(s.size(1));
  const c10::cuda::CUDAGuard guard(s.device());
  const bool y_bf16 = dy.scalar_type() == torch::kBFloat16;
  const bool has_ds = ds_in.has_value() && ds_in->defined();
  if (has_ds) {
    check_input(*ds_in, "ds_in");
    TORCH_CHECK(ds_in->scalar_type() == s.scalar_type(), "ds_in must have the dtype of s");
  }
  TORCH_CHECK(s.scalar_type() == st(a_bf16 && r_bf16), "s dtype does not match a and r");

  auto opts = s.options();
  auto da = torch::empty({N, D}, opts.dtype(st(a_bf16)));
  auto dr = torch::empty({N, D}, opts.dtype(st(r_bf16)));
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t P = std::max<int64_t>(1, std::min<int64_t>(4 * sms, N));
  auto dw_partial = torch::empty({P, D}, opts.dtype(torch::kFloat32));
  if (N == 0) return {da, dr, dw_partial.zero_()};

  int threads, vpt;
  launch_shape(D, threads, vpt);
  auto stream = at::cuda::getCurrentCUDAStream();
  dispatch(a_bf16, r_bf16, y_bf16, [&](auto ta, auto tr, auto ty) {
    using TA = decltype(ta);
    using TR = decltype(tr);
    using TY = decltype(ty);
    using TS = wider_t<TA, TR>;
    const TS* dsp = has_ds ? ptr<TS>(*ds_in) : ptr<TS>(s);
    auto args = std::make_tuple(ptr<TY>(dy), dsp, ptr<TS>(s), ptr<float>(w), ptr<float>(rstd),
                                ptr<TA>(da), ptr<TR>(dr), ptr<float>(dw_partial), N, D, has_ds);
    auto go = [&](auto kernel) {
      std::apply([&](auto... xs) { kernel<<<P, threads, 0, stream>>>(xs...); }, args);
    };
    if (vpt == 1) go(add_rmsnorm_bwd_kernel<TA, TR, TS, TY, 1>);
    else if (vpt == 2) go(add_rmsnorm_bwd_kernel<TA, TR, TS, TY, 2>);
    else if (vpt == 4) go(add_rmsnorm_bwd_kernel<TA, TR, TS, TY, 4>);
    else go(add_rmsnorm_bwd_kernel<TA, TR, TS, TY, 8>);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {da, dr, dw_partial};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "fused add + RMSNorm forward (CUDA)");
  m.def("backward", &backward, "fused add + RMSNorm backward (CUDA)");
}
