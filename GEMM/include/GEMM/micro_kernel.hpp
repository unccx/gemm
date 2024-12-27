#pragma once

#include <GEMM/matrix_view.hpp>
#include <cassert>
#include <cstddef>
namespace GEMM {

/**
 * @brief
 *
 * @tparam T
 * @tparam layout 输出矩阵的内存布局
 * @tparam GEMM_MR
 * @tparam GEMM_NR
 */
template <typename T, StorageLayout layout, size_t GEMM_MR, size_t GEMM_NR>
class MicroKernel {
public:
  typedef T type;

  static void block_update(const MatrixView<T, StorageLayout::ColumnMajor> A,
                           const MatrixView<T, StorageLayout::RowMajor> B,
                           MatrixView<T, layout> C);
};

template <typename T, StorageLayout layout, size_t GEMM_MR, size_t GEMM_NR>
void MicroKernel<T, layout, GEMM_MR, GEMM_NR>::block_update(
    const MatrixView<T, StorageLayout::ColumnMajor> A,
    const MatrixView<T, StorageLayout::RowMajor> B, MatrixView<T, layout> C) {
  assert(A.cols == B.rows);
  const size_t k_c = A.cols;

  for (size_t l = 0; l < k_c; l++) {
    if constexpr (layout == StorageLayout::ColumnMajor) {
      for (size_t j = 0; j < GEMM_NR; j++) {
        for (size_t i = 0; i < GEMM_MR; i++) {
          C(i, j) += A(i, l) * B(l, j);
        }
      }
    } else {
      for (size_t i = 0; i < GEMM_MR; i++) {
        for (size_t j = 0; j < GEMM_NR; j++) {
          C(i, j) += A(i, l) * B(l, j);
        }
      }
    }
  }
}

} // namespace GEMM