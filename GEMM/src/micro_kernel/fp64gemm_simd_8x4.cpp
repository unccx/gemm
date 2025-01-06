#include <GEMM/micro_kernel.hpp>
#include <cassert>
#include <cstddef>
#include <immintrin.h>

#define _MM_SHUFFLER(w, x, y, z) _MM_SHUFFLE(z, y, x, w)

namespace GEMM {
constexpr size_t GEMM_MR = 8;
constexpr size_t GEMM_NR = 4;
using fp64_mirco_kernel =
    SIMDMicroKernel<double, StorageLayout::ColumnMajor, GEMM_MR, GEMM_NR>;

template class SIMDMicroKernel<double, StorageLayout::ColumnMajor, GEMM_MR,
                               GEMM_NR>;

template <>
void fp64_mirco_kernel::block_update(
    const MatrixView<double, StorageLayout::ColumnMajor> A,
    const MatrixView<double, StorageLayout::RowMajor> B,
    MatrixView<double, StorageLayout::ColumnMajor> C) {
  assert(A.rows == GEMM_MR);
  assert(A.cols == B.rows);
  assert(B.cols == GEMM_NR);
  const size_t k_c = A.cols;
  constexpr size_t unroll_factor = 2;
  constexpr size_t vlen = 256 / 64;

  double *a = A.data;
  double *b = B.data;
  double *c = C.data;

  _mm_prefetch(A.data, _MM_HINT_T0);
  _mm_prefetch(fp64_mirco_kernel::next_B_panel, _MM_HINT_T2);
  _mm_prefetch(C.data, _MM_HINT_T0);

  auto vc0_3__0 = _mm256_setzero_pd(); // c[0:3][0]
  auto vc0_3__1 = _mm256_setzero_pd(); // c[0:3][1]
  auto vc0_3__2 = _mm256_setzero_pd(); // c[0:3][2]
  auto vc0_3__3 = _mm256_setzero_pd(); // c[0:3][3]

  auto vc4_7__0 = _mm256_setzero_pd(); // c[4:7][0]
  auto vc4_7__1 = _mm256_setzero_pd(); // c[4:7][1]
  auto vc4_7__2 = _mm256_setzero_pd(); // c[4:7][2]
  auto vc4_7__3 = _mm256_setzero_pd(); // c[4:7][3]

  // ping 缓冲
  __m256d ping_va0_3 = _mm256_loadu_pd(a);         // load a[0:3]
  __m256d ping_va4_7 = _mm256_loadu_pd(a += vlen); // load a[4:7]
  __m256d ping_vb0_3 = _mm256_loadu_pd(b);         // load vb0_3 (b0,b1,b2,b3)

  // pong 缓冲
  __m256d pong_va0_3;
  __m256d pong_va4_7;
  __m256d pong_vb0_3;

  // load vb0_3 (b0,b1,b2,b3)
  // Shuffle vb (b1,b0,b3,b2)
  // Permute vb (b3,b2,b1,b0)
  // Shuffle vb (b2,b3,b0,b1)

  // AVX 8×4 rank-1 update with butterfly permutation.
  for (size_t i = 0; i < k_c / unroll_factor; i++) {
    // Iteration 0.——————————————————————————————————————————————————————

    // Prefetch a[0:3] into pong buffer for next iteration
    pong_va0_3 = _mm256_loadu_pd(a += vlen); // Latency = 7

    // Compute C[0:7][0]
    vc0_3__0 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__0); // Latency = 4
    vc4_7__0 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__0); // Latency = 4

    // Prefetch a[4:7] into pong buffer for next iteration
    pong_va4_7 = _mm256_loadu_pd(a += vlen); // Latency = 7

    // Shuffle vb (b1,b0,b3,b2)
    ping_vb0_3 = _mm256_shuffle_pd(ping_vb0_3, ping_vb0_3, 0x5); // Latency = 1

    // Compute C[0:7][1]
    vc0_3__1 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__1); // Latency = 4
    vc4_7__1 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__1); // Latency = 4

    // Permute vb (b3,b2,b1,b0)
    ping_vb0_3 =
        _mm256_permute2f128_pd(ping_vb0_3, ping_vb0_3, 0x1); // Latency = 3

    // Prefetch a[4:7] into pong buffer for next iteration
    pong_vb0_3 = _mm256_loadu_pd(b += vlen); // Latency = 7

    // Compute C[0:7][2]
    vc0_3__2 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__2); // Latency = 4
    vc4_7__2 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__2); // Latency = 4

    // Shuffle vb (b2,b3,b0,b1)
    ping_vb0_3 = _mm256_shuffle_pd(ping_vb0_3, ping_vb0_3, 0x5); // Latency = 1

    // Compute C[0:7][3]
    vc0_3__3 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__3); // Latency = 4
    vc4_7__3 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__3); // Latency = 4

    // Iteration 1.——————————————————————————————————————————————————————

    // Prefetch a[0:3] into ping buffer for next iteration
    ping_va0_3 = _mm256_loadu_pd(a += vlen); // Latency = 7

    // Compute C[0:7][0]
    vc0_3__0 = _mm256_fmadd_pd(pong_va0_3, pong_vb0_3, vc0_3__0); // Latency = 4
    vc4_7__0 = _mm256_fmadd_pd(pong_va4_7, pong_vb0_3, vc4_7__0); // Latency = 4

    // Shuffle vb (b1,b0,b3,b2)
    pong_vb0_3 = _mm256_shuffle_pd(pong_vb0_3, pong_vb0_3, 0x5); // Latency = 1

    // Compute C[0:7][1]
    vc0_3__1 = _mm256_fmadd_pd(pong_va0_3, pong_vb0_3, vc0_3__1); // Latency = 4
    vc4_7__1 = _mm256_fmadd_pd(pong_va4_7, pong_vb0_3, vc4_7__1); // Latency = 4

    // Prefetch a[4:7] into ping buffer for next iteration
    ping_va4_7 = _mm256_loadu_pd(a += vlen); // Latency = 7

    // Permute vb (b3,b2,b1,b0)
    pong_vb0_3 =
        _mm256_permute2f128_pd(pong_vb0_3, pong_vb0_3, 0x1); // Latency = 3

    // Compute C[0:7][2]
    vc0_3__2 = _mm256_fmadd_pd(pong_va0_3, pong_vb0_3, vc0_3__2); // Latency = 4
    vc4_7__2 = _mm256_fmadd_pd(pong_va4_7, pong_vb0_3, vc4_7__2); // Latency = 4

    // Shuffle vb (b3,b2,b1,b0)
    pong_vb0_3 = _mm256_shuffle_pd(pong_vb0_3, pong_vb0_3, 0x5); // Latency = 1

    // Prefetch a[4:7] into pong buffer for next iteration
    ping_vb0_3 = _mm256_loadu_pd(b += vlen); // Latency = 7

    // Compute C[0:7][3]
    vc0_3__3 = _mm256_fmadd_pd(pong_va0_3, pong_vb0_3, vc0_3__3); // Latency = 4
    vc4_7__3 = _mm256_fmadd_pd(pong_va4_7, pong_vb0_3, vc4_7__3); // Latency = 4
  }

  // handle loop unroll tail data
  const size_t tail = k_c % unroll_factor;
  for (size_t i = 0; i < tail; i++) {
    // Iteration 0.——————————————————————————————————————————————————————

    // Compute C[0:7][0]
    vc0_3__0 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__0); // Latency = 4
    vc4_7__0 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__0); // Latency = 4

    // Shuffle vb (b1,b0,b3,b2)
    ping_vb0_3 = _mm256_shuffle_pd(ping_vb0_3, ping_vb0_3, 0x5); // Latency = 1

    // Compute C[0:7][1]
    vc0_3__1 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__1); // Latency = 4
    vc4_7__1 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__1); // Latency = 4

    // Permute vb (b3,b2,b1,b0)
    ping_vb0_3 =
        _mm256_permute2f128_pd(ping_vb0_3, ping_vb0_3, 0x1); // Latency = 3

    // Compute C[0:7][2]
    vc0_3__2 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__2); // Latency = 4
    vc4_7__2 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__2); // Latency = 4

    // Shuffle vb (b3,b2,b1,b0)
    ping_vb0_3 = _mm256_shuffle_pd(ping_vb0_3, ping_vb0_3, 0x5); // Latency = 1

    // Compute C[0:7][3]
    vc0_3__3 = _mm256_fmadd_pd(ping_va0_3, ping_vb0_3, vc0_3__3); // Latency = 4
    vc4_7__3 = _mm256_fmadd_pd(ping_va4_7, ping_vb0_3, vc4_7__3); // Latency = 4
  }

  // swap element 将内存中连续的元素置换到连续的向量中
  {
    __m256d vtmpa_0_3b_0 = _mm256_blend_pd(vc0_3__0, vc0_3__1, 0x6);
    __m256d vtmpa_0_3b_1 = _mm256_blend_pd(vc0_3__1, vc0_3__0, 0x6);

    __m256d vtmpa_0_3b_2 = _mm256_blend_pd(vc0_3__2, vc0_3__3, 0x6);
    __m256d vtmpa_0_3b_3 = _mm256_blend_pd(vc0_3__3, vc0_3__2, 0x6);

    __m256d vtmpa_4_7b_0 = _mm256_blend_pd(vc4_7__0, vc4_7__1, 0x6);
    __m256d vtmpa_4_7b_1 = _mm256_blend_pd(vc4_7__1, vc4_7__0, 0x6);

    __m256d vtmpa_4_7b_2 = _mm256_blend_pd(vc4_7__2, vc4_7__3, 0x6);
    __m256d vtmpa_4_7b_3 = _mm256_blend_pd(vc4_7__3, vc4_7__2, 0x6);

    vc0_3__0 = _mm256_permute2f128_pd(vtmpa_0_3b_0, vtmpa_0_3b_2, 0x30);
    vc0_3__3 = _mm256_permute2f128_pd(vtmpa_0_3b_2, vtmpa_0_3b_0, 0x30);

    vc0_3__1 = _mm256_permute2f128_pd(vtmpa_0_3b_1, vtmpa_0_3b_3, 0x30);
    vc0_3__2 = _mm256_permute2f128_pd(vtmpa_0_3b_3, vtmpa_0_3b_1, 0x30);

    vc4_7__0 = _mm256_permute2f128_pd(vtmpa_4_7b_0, vtmpa_4_7b_2, 0x30);
    vc4_7__3 = _mm256_permute2f128_pd(vtmpa_4_7b_2, vtmpa_4_7b_0, 0x30);

    vc4_7__1 = _mm256_permute2f128_pd(vtmpa_4_7b_1, vtmpa_4_7b_3, 0x30);
    vc4_7__2 = _mm256_permute2f128_pd(vtmpa_4_7b_3, vtmpa_4_7b_1, 0x30);
  }

  // save into C
  if (C.rows == GEMM_MR && C.cols == GEMM_NR) {
    auto C00 = c + vlen * 0;
    vc0_3__0 = _mm256_add_pd(vc0_3__0, _mm256_loadu_pd(C00));
    _mm256_storeu_pd(C00, vc0_3__0);

    auto C40 = c + vlen * 1;
    vc4_7__0 = _mm256_add_pd(vc4_7__0, _mm256_loadu_pd(C40));
    _mm256_storeu_pd(C40, vc4_7__0);

    auto C01 = c + vlen * 2;
    vc0_3__1 = _mm256_add_pd(vc0_3__1, _mm256_loadu_pd(C01));
    _mm256_storeu_pd(C01, vc0_3__1);

    auto C41 = c + vlen * 3;
    vc4_7__1 = _mm256_add_pd(vc4_7__1, _mm256_loadu_pd(C41));
    _mm256_storeu_pd(C41, vc4_7__1);

    auto C02 = c + vlen * 4;
    vc0_3__2 = _mm256_add_pd(vc0_3__2, _mm256_loadu_pd(C02));
    _mm256_storeu_pd(C02, vc0_3__2);

    auto C42 = c + vlen * 5;
    vc4_7__2 = _mm256_add_pd(vc4_7__2, _mm256_loadu_pd(C42));
    _mm256_storeu_pd(C42, vc4_7__2);

    auto C03 = c + vlen * 6;
    vc0_3__3 = _mm256_add_pd(vc0_3__3, _mm256_loadu_pd(C03));
    _mm256_storeu_pd(C03, vc0_3__3);

    auto C43 = c + vlen * 7;
    vc4_7__3 = _mm256_add_pd(vc4_7__3, _mm256_loadu_pd(C43));
    _mm256_storeu_pd(C43, vc4_7__3);
  } else { // if C.rows < GEMM_MR or C.cols < GEMM_NR
    // save edge data
    double buf[GEMM_MR * GEMM_NR] = {0};
    _mm256_storeu_pd(buf + vlen * 0, vc0_3__0);
    _mm256_storeu_pd(buf + vlen * 1, vc4_7__0);
    _mm256_storeu_pd(buf + vlen * 2, vc0_3__1);
    _mm256_storeu_pd(buf + vlen * 3, vc4_7__1);
    _mm256_storeu_pd(buf + vlen * 4, vc0_3__2);
    _mm256_storeu_pd(buf + vlen * 5, vc4_7__2);
    _mm256_storeu_pd(buf + vlen * 6, vc0_3__3);
    _mm256_storeu_pd(buf + vlen * 7, vc4_7__3);
    MatrixView<double, StorageLayout::ColumnMajor> buffer{GEMM_MR, GEMM_NR,
                                                          buf};
    for (size_t j = 0; j < C.cols; j++) {
      for (size_t i = 0; i < C.rows; i++) {
        C(i, j) += buffer(i, j);
      }
    }
  }
}
} // namespace GEMM
