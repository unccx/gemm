#pragma once
#include <GEMM/matrix_view.hpp>
#include <GEMM/micro_kernel.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace GEMM {
/**
 * @brief 把矩阵按照行主序或列主序 pack into 到 packed_panel
 *
 * @tparam T
 * @tparam layout
 * @tparam stride
 * @param sub_mat
 * @param packed_panel 保存 packed data 的指针
 *                     需要调用者保证指针所持有的内存size>=stride*k_c
 */
template <typename T, StorageLayout mat_layout, size_t stride,
          StorageLayout panel_layout>
MatrixView<T, panel_layout> pack_panel(const MatrixView<T, mat_layout> sub_mat,
                                       T *packed_panel) {
  auto pack_row_major = [&]() {
    MatrixView<T, panel_layout> ret{sub_mat.rows, stride, packed_panel};
    for (size_t i = 0; i < sub_mat.rows; i++) {
      for (size_t j = 0; j < sub_mat.cols; j++) {
        *(packed_panel++) = sub_mat.get(i, j);
      }
      for (size_t j = sub_mat.cols; j < stride; j++) {
        *(packed_panel++) = sub_mat.get(i, 0);
      }
    }
    return ret;
  };

  auto pack_col_major = [&]() {
    MatrixView<T, panel_layout> ret{stride, sub_mat.cols, packed_panel};
    for (size_t j = 0; j < sub_mat.cols; j++) {
      for (size_t i = 0; i < sub_mat.rows; i++) {
        *(packed_panel++) = sub_mat.get(i, j);
      }
      for (size_t i = sub_mat.rows; i < stride; i++) {
        *(packed_panel++) = sub_mat.get(0, j);
      }
    }
    return ret;
  };

  if constexpr (panel_layout == StorageLayout::RowMajor) {
    return pack_row_major();
  } else {
    return pack_col_major();
  }
}

/**
 * @brief
 *
 * @tparam T
 * @tparam GEMM_MR
 * @tparam GEMM_NR
 * @tparam MICRO_KERNEL
 * @param m
 * @param k
 * @param n
 * @param packA
 * @param packB
 * @param C
 */
template <typename T, StorageLayout layout, size_t GEMM_MR, size_t GEMM_NR,
          typename MICRO_KERNEL>
void macro_kernel(const size_t m, const size_t k, const size_t n, T *packA,
                  T *packB, MatrixView<T, layout> C) {
  // 2-th loop around micro-kernel
  for (size_t j = 0; j < n; j += GEMM_NR) { // j 是 B_panel 中的列索引
    // 1-th loop around micro-kernel
    for (size_t i = 0; i < m; i += GEMM_MR) { // i 是 A_panel 中的行索引
      MatrixView<T, StorageLayout::ColumnMajor> A_micro_panel{GEMM_MR, k,
                                                              &packA[i * k]};
      MatrixView<T, StorageLayout::RowMajor> B_micro_panel{k, GEMM_NR,
                                                           &packB[j * k]};
      MICRO_KERNEL::block_update(A_micro_panel, B_micro_panel,
                                 C.submat(i, j, GEMM_MR, GEMM_NR));
    } // 1-th loop around micro-kernel
  } // 2-th loop around micro-kernel
}

/**
 * @brief 通用矩阵-矩阵乘法 A{m,k} x B{k,n} = C{m,n} ,矩阵元素为 T
 *
 * @tparam T 矩阵元素类型
 * @tparam GEMM_NC 第一次分块 GEMP 的 B 矩阵分块粒度
 * @tparam GEMM_KC 第二次分块 GEPP 的 A 和 B 矩阵分块粒度
 * @tparam GEMM_MC 第一次分块 GEPM 的 A 矩阵分块粒度
 * @tparam GEMM_MR
 * @tparam GEMM_NR
 * @tparam MICRO_KERNEL
 * @tparam GEMM_SIMD_ALIGN_SIZE
 * @param A 输入矩阵
 * @param B 输入矩阵
 * @return MatrixView<T> 输出矩阵
 */
template <typename T, StorageLayout layout = StorageLayout::ColumnMajor,
          size_t GEMM_NC = 4080, size_t GEMM_KC = 256, size_t GEMM_MC = 72,
          size_t GEMM_MR = 8, size_t GEMM_NR = 4,
          typename MICRO_KERNEL = MicroKernel<T, layout, GEMM_MR, GEMM_NR>,
          size_t GEMM_SIMD_ALIGN_SIZE = 32>
MatrixView<T, layout> gemm(const MatrixView<T, layout> A,
                           const MatrixView<T, layout> B) {
  // check shape
  assert(A.cols == B.rows);
  const auto &m = A.rows;
  const auto &k = A.cols; // B.rows
  const auto &n = B.cols;
  if (m == 0 || k == 0 || n == 0) {
    return {0, 0, nullptr};
  }

  // Allocate C matrix buffer
  T *C_data = (T *)std::aligned_alloc(GEMM_SIMD_ALIGN_SIZE, m * n * sizeof(T));
  MatrixView<T, StorageLayout::ColumnMajor> C{m, n, C_data};

  // Allocate packing buffers
  T *packA = (T *)std::aligned_alloc(GEMM_SIMD_ALIGN_SIZE,
                                     (GEMM_MC + GEMM_MR) * GEMM_KC * sizeof(T));
  T *packB = (T *)std::aligned_alloc(GEMM_SIMD_ALIGN_SIZE,
                                     GEMM_KC * (GEMM_NC + GEMM_NR) * sizeof(T));

  // 5-th loop around micro-kernel
  for (size_t j_c = 0; j_c < n; j_c += GEMM_NC) {
    size_t n_c = std::min(n - j_c, GEMM_NC);

    // 4-th loop around micro-kernel
    for (size_t p_c = 0; p_c < k; p_c += GEMM_KC) {
      size_t k_c = std::min(k - p_c, GEMM_KC);
      // pack into B_panel
      for (size_t j = 0; j < n_c; j += GEMM_NR) { // j 是 B_panel 中的列索引
        size_t n_r = std::min(n_c - j, GEMM_NR);
        pack_panel<T, layout, GEMM_NR, StorageLayout::RowMajor>(
            B.submat(p_c, j_c + j, k_c, n_r), &packB[j * k_c]);
      }

      // 3-rd loop around micro-kernel
      for (size_t i_c = 0; i_c < m; i_c += GEMM_MC) {
        size_t m_c = std::min(m - i_c, GEMM_MC);
        // pack into A_panel
        for (size_t i = 0; i < m_c; i += GEMM_MR) { // i 是 A_panel 中的行索引
          size_t m_r = std::min(m_c - i, GEMM_MR);
          pack_panel<T, layout, GEMM_MR, StorageLayout::ColumnMajor>(
              A.submat(i_c + i, p_c, m_r, k_c), &packA[i * k_c]);
        }

        macro_kernel<T, layout, GEMM_MR, GEMM_NR, MICRO_KERNEL>(
            m_c, k_c, n_c, packA, packB, C.submat(i_c, j_c, m_c, n_c));
      } // End 3.rd loop around micro-kernel
    } // End 4.th loop around micro-kernel
  } // End 5.th loop around micro-kernel

  std::free(packA);
  std::free(packB);

  return C;
}

} // namespace GEMM