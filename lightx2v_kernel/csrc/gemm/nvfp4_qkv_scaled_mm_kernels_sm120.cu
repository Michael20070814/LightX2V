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

struct Fp4QkvGemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = cutlass::bfloat16_t;
  using ElementC = cutlass::bfloat16_t;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
#if defined(LIGHTX2V_THOR_NVFP4_ONLY)
  using ArchTag = cutlass::arch::Sm100;
#else
  using ArchTag = cutlass::arch::Sm120;
#endif
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ThreadBlockShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  using EVTOp = cutlass::epilogue::fusion::PerColLinCombPerColBiasEltAct<
      cutlass::epilogue::thread::Identity,
      ElementD,
      ElementAccumulator>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ThreadBlockShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutCTag,
      AlignmentC,
      ElementD,
      LayoutDTag,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
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
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
};

typename Fp4QkvGemmSm120::Gemm::Arguments make_fp4_qkv_gemm_arguments(
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
  using Sm1xxBlkScaledConfig =
      typename Fp4QkvGemmSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(Fp4QkvGemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Fp4QkvGemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Fp4QkvGemmSm120::StrideD{}, {m, n, 1});
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  typename Fp4QkvGemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {
          static_cast<Fp4QkvGemmSm120::Gemm::ElementA const*>(A.data_ptr()),
          stride_A,
          static_cast<Fp4QkvGemmSm120::Gemm::ElementB const*>(B.data_ptr()),
          stride_B,
          static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
          layout_SFA,
          static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
          layout_SFB,
      },
      {
          {},
          static_cast<Fp4QkvGemmSm120::Gemm::ElementC const*>(D.data_ptr()),
          stride_D,
          static_cast<Fp4QkvGemmSm120::Gemm::ElementD*>(D.data_ptr()),
          stride_D,
      },
  };

  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
  fusion_args.dAlpha = {_0{}, true, 0};
  fusion_args.beta = 0.0f;
  fusion_args.dBeta = {_0{}, false, 0};
  if (bias) {
    fusion_args.bias_ptr =
        static_cast<Fp4QkvGemmSm120::Gemm::ElementC const*>(bias->data_ptr());
  }
  return arguments;
}

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
  typename Fp4QkvGemmSm120::Gemm gemm;
  auto arguments = make_fp4_qkv_gemm_arguments(D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  size_t workspace_size = Fp4QkvGemmSm120::Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      workspace_size,
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

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
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(
      A.sizes()[1] == B.sizes()[1],
      "a and b shapes cannot be multiplied (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      " and ",
      B.sizes()[0],
      "x",
      B.sizes()[1],
      ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;

  TORCH_CHECK(A.get_device() == B.get_device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == A_sf.get_device(), "a and scale_a must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == B_sf.get_device(), "a and scale_b must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == alpha.get_device(), "a and alpha must be on the same CUDA device");
  TORCH_CHECK(A.get_device() == D.get_device(), "a and out must be on the same CUDA device");
  TORCH_CHECK(alpha.numel() == n, "alpha must contain one value per output column (", n, ")");
  if (bias) {
    auto const& bias_tensor = *bias;
    CHECK_INPUT(bias_tensor, at::ScalarType::BFloat16, "bias");
    TORCH_CHECK(A.get_device() == bias_tensor.get_device(), "a and bias must be on the same CUDA device");
    TORCH_CHECK(bias_tensor.numel() == n, "bias must contain one value per output column (", n, ")");
  }
  TORCH_CHECK(D.sizes() == torch::IntArrayRef({m, n}), "out must have shape (", m, ", ", n, ")");

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
      B_sf.sizes() == torch::IntArrayRef({rounded_n, rounded_k}),
      "scale_b must have shape (",
      rounded_n,
      ", ",
      rounded_k,
      ")");

  at::cuda::CUDAGuard device_guard{A.device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  run_fp4_qkv_gemm_sm120(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
}
