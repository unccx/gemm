#include <GEMM/gemm.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace GEMM {

template <typename T, size_t STRIDE>
void pack_panel(size_t cols, size_t rows, T *data, size_t ld, size_t offset,
                T *packed_panel) {
  T *ptrs[STRIDE];

  for (size_t i = 0; i < cols; i++) {
    ptrs[i] = data + ld * (offset + i);
  }

  // 用于处理 cols < STRIDE 的 tail data 通过重复第一列的数据进行 padding
  for (size_t i = cols; i < STRIDE; i++) {
    ptrs[i] = data + ld * (offset + 0);
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < STRIDE; j++) {
      *(packed_panel++) = *(ptrs[j]++);
    }
  }
}

template <typename T, size_t GEMM_KC, size_t GEMM_NC, size_t GEMM_MC,
          size_t GEMM_MR, size_t GEMM_NR, typename MICRO_KERNEL,
          size_t GEMM_SIMD_ALIGN_SIZE>
void gemm(size_t m, size_t k, size_t n, T *A, size_t lda, T *B, size_t ldb,
          T *C, size_t ldc) {
  // check shape
  if (m == 0 || k == 0 || k == 0) {
    return;
  }

  T *packA = (T *)std::aligned_alloc(GEMM_SIMD_ALIGN_SIZE, GEMM_MC * GEMM_KC);
  T *packB = (T *)std::aligned_alloc(GEMM_SIMD_ALIGN_SIZE, GEMM_KC * GEMM_MC);

  for (size_t j_c = 0; j_c < n; j_c += GEMM_NC) {
    size_t n_c = std::min(n - j_c, GEMM_NC);
    for (size_t p_c = 0; p_c < k; p_c += GEMM_KC) {
      size_t k_c = std::min(n - p_c, GEMM_NC);

      // pack into B_panel
      for (size_t j = 0; j < n_c; j += GEMM_NR) {
        size_t n_r = std::min(n_c - j, GEMM_NR);
        pack_panel<T, GEMM_NR>(n_r, k_c);
      }
    }
  }
}

} // namespace GEMM