#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

#define CHECK_TYPE(x, st, m) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

using namespace cute;

constexpr int kQkvBatchCount = 3;
constexpr int kQkvSharedMemoryLimitBytes = 114 * 1024;

template <bool HasBias>
struct Fp4QkvGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = cutlass::bfloat16_t;
  using ElementC = void;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 1;

  using ElementAccumulator = float;
#if defined(LIGHTX2V_THOR_NVFP4_ONLY)
  using ArchTag = cutlass::arch::Sm100;
#else
  using ArchTag = cutlass::arch::Sm120;
#endif
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ThreadBlockShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
#if defined(LIGHTX2V_THOR_NVFP4_ONLY)
  using EpilogueTile = Shape<_128, _32>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
#else
  using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
#endif

  using EVTOp = cute::conditional_t<
      HasBias,
      cutlass::epilogue::fusion::LinCombPerColBias<
          ElementD,
          ElementAccumulator,
          ElementD,
          ElementC,
          ElementAccumulator>,
      cutlass::epilogue::fusion::ScaledAcc<ElementD, ElementAccumulator, ElementAccumulator>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ThreadBlockShape,
      ClusterShape,
      EpilogueTile,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      EpilogueSchedule,
      EVTOp>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutATag,
      AlignmentA,
      ElementB,
      LayoutBTag,
      AlignmentB,
      ElementAccumulator,
      ThreadBlockShape,
      ClusterShape,
      cutlass::gemm::collective::StageCount<2>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernelBase = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  struct GemmKernel : GemmKernelBase {
    static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
  };

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static_assert(
      GemmKernel::SharedStorageSize <= kQkvSharedMemoryLimitBytes,
      "Fused QKV dynamic shared memory exceeds the 2 CTA/SM budget");

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

template <bool HasBias>
typename Fp4QkvGemmSm120<HasBias>::Gemm::Arguments make_fp4_qkv_gemm_arguments(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t M,
    int64_t N,
    int64_t K) {
  using KernelConfig = Fp4QkvGemmSm120<HasBias>;
  using Sm1xxBlkScaledConfig =
      typename KernelConfig::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto problem_shape = cute::make_shape(m, n, k, kQkvBatchCount);
  auto stride_A = cutlass::make_cute_packed_stride(
      typename KernelConfig::StrideA{}, cute::make_shape(m, k, kQkvBatchCount));
  auto stride_B = cutlass::make_cute_packed_stride(
      typename KernelConfig::StrideB{}, cute::make_shape(n, k, kQkvBatchCount));
  auto stride_D = cutlass::make_cute_packed_stride(
      typename KernelConfig::StrideD{}, cute::make_shape(m, n, kQkvBatchCount));

  // Q, K, and V share the same activation and activation scale-factor storage.
  cute::get<2>(stride_A) = 0;
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape);
  cute::get<1>(cute::get<2>(layout_SFA.stride())) = 0;

  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape);

  // Interleave Q/K/V within each output row so D is contiguous when viewed as [M, 3N].
  cute::get<0>(stride_D) = static_cast<int64_t>(kQkvBatchCount) * n;
  cute::get<2>(stride_D) = static_cast<int64_t>(n);

  typename KernelConfig::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {
          static_cast<typename KernelConfig::Gemm::ElementA const*>(A.data_ptr()),
          stride_A,
          static_cast<typename KernelConfig::Gemm::ElementB const*>(B.data_ptr()),
          stride_B,
          static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
          layout_SFA,
          static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
          layout_SFB,
      },
      {
          {},
          nullptr,
          typename KernelConfig::StrideC{},
          static_cast<typename KernelConfig::Gemm::ElementD*>(D.data_ptr()),
          stride_D,
      },
  };

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
  fusion_args.dAlpha = {_0{}, _0{}, 1};
  if constexpr (HasBias) {
    TORCH_INTERNAL_ASSERT(bias.has_value());
    fusion_args.beta = 0.0f;
    fusion_args.bias_ptr =
        static_cast<typename KernelConfig::Gemm::ElementD const*>(bias->data_ptr());
    fusion_args.dBias = {_0{}, _1{}, static_cast<int64_t>(n)};
  }
  return arguments;
}

template <bool HasBias>
void run_fp4_qkv_gemm_sm120(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  using KernelConfig = Fp4QkvGemmSm120<HasBias>;
  typename KernelConfig::Gemm gemm;
  auto arguments = make_fp4_qkv_gemm_arguments<HasBias>(D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  size_t workspace_size = KernelConfig::Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      workspace_size,
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

  TORCH_CHECK(
      KernelConfig::Gemm::GemmKernel::SharedStorageSize <= kQkvSharedMemoryLimitBytes,
      "Fused QKV dynamic shared memory is ",
      KernelConfig::Gemm::GemmKernel::SharedStorageSize,
      " bytes, exceeding the ",
      kQkvSharedMemoryLimitBytes,
      " byte 2 CTA/SM limit");
  int max_shared_memory_per_sm = 0;
  auto device_attribute_status = cudaDeviceGetAttribute(
      &max_shared_memory_per_sm,
      cudaDevAttrMaxSharedMemoryPerMultiprocessor,
      A.get_device());
  TORCH_CHECK(
      device_attribute_status == cudaSuccess,
      "Failed to query per-SM shared memory: ",
      cudaGetErrorString(device_attribute_status));
  TORCH_CHECK(
      2 * KernelConfig::Gemm::GemmKernel::SharedStorageSize <= max_shared_memory_per_sm,
      "Fused QKV needs ",
      2 * KernelConfig::Gemm::GemmKernel::SharedStorageSize,
      " bytes of shared memory for 2 CTA/SM, but the device provides ",
      max_shared_memory_per_sm,
      " bytes per SM");
  auto carveout_status = cudaFuncSetAttribute(
      cutlass::device_kernel<typename KernelConfig::Gemm::GemmKernel>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100);
  TORCH_CHECK(
      carveout_status == cudaSuccess,
      "Failed to select the 100% shared-memory carveout: ",
      cudaGetErrorString(carveout_status));
  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void cutlass_scaled_nvfp4_qkv_mm_sm120(
    torch::Tensor& D,
    torch::Tensor const& A,
    torch::Tensor const& B,
    torch::Tensor const& A_sf,
    torch::Tensor const& B_sf,
    torch::Tensor const& alpha,
    c10::optional<torch::Tensor> const& bias) {
  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");
  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");
  CHECK_INPUT(D, at::ScalarType::BFloat16, "out");

  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 3 && B.sizes()[0] == kQkvBatchCount, "b must have shape (3, N, K / 2)");
  TORCH_CHECK(
      A.sizes()[1] == B.sizes()[2],
      "a and b shapes cannot be multiplied (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      " and 3x",
      B.sizes()[1],
      "x",
      B.sizes()[2],
      ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[1];
  auto const k = A.sizes()[1] * 2;

  TORCH_CHECK(A.get_device() == B.get_device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == A_sf.get_device(), "a and scale_a must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == B_sf.get_device(), "a and scale_b must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == alpha.get_device(), "a and alpha must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == D.get_device(), "a and out must be on the same CUDA device");
  TORCH_CHECK(alpha.sizes() == torch::IntArrayRef({kQkvBatchCount}), "alpha must have shape (3,)");
  if (bias) {
    auto const& bias_tensor = *bias;
    CHECK_INPUT(bias_tensor, at::ScalarType::BFloat16, "bias");
    TORCH_CHECK(A.get_device() == bias_tensor.get_device(), "a and bias must be on the same CUDA device");
    TORCH_CHECK(
        bias_tensor.sizes() == torch::IntArrayRef({kQkvBatchCount, n}),
        "bias must have shape (3, ",
        n,
        ")");
  }
  TORCH_CHECK(
      D.sizes() == torch::IntArrayRef({m, kQkvBatchCount, n}),
      "out must have shape (",
      m,
      ", 3, ",
      n,
      ")");

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(
      A_sf.sizes() == torch::IntArrayRef({rounded_m, rounded_k}),
      "scale_a must have shape (",
      rounded_m,
      ", ",
      rounded_k,
      ")");
  TORCH_CHECK(
      B_sf.sizes() == torch::IntArrayRef({kQkvBatchCount, rounded_n, rounded_k}),
      "scale_b must have shape (3, ",
      rounded_n,
      ", ",
      rounded_k,
      ")");

  at::cuda::CUDAGuard device_guard{A.device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  if (bias) {
    run_fp4_qkv_gemm_sm120<true>(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
  } else {
    run_fp4_qkv_gemm_sm120<false>(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
  }
}
