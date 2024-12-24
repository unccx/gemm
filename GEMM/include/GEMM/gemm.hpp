#pragma once
#include <GEMM/micro_kernel.hpp>
#include <cassert>
#include <cstddef>

namespace GEMM {
enum class StorageLayout {
  RowMajor = 0,   // 行优先
  ColumnMajor = 1 // 列优先
};

template <typename T, StorageLayout layout = StorageLayout::RowMajor>
class MatrixView {
public:
  size_t start_row, start_col;
  size_t rows, cols;
  size_t lda;
  T *data;

  /**
   * @brief Construct a new Matrix View object
   *
   * @param rows 行数
   * @param cols 列数
   * @param data 指向数据的指针
   * @param lda  leading dimension, 默认为 0
   *             若为 0 则根据 RowMajor 或 ColumnMajor 选择 cols 或 rows
   * @param start_row submatrix窗口的左上角坐标行
   * @param start_col submatrix窗口的左上角坐标列
   */
  MatrixView(size_t rows, size_t cols, T *data, size_t lda = 0,
             size_t start_row = 0, size_t start_col = 0)
      : rows(rows), cols(cols), data(data), start_row(start_row),
        start_col(start_col) {
    if constexpr (layout == StorageLayout::RowMajor) {
      this->lda = lda > 0 ? lda : cols;
    } else {
      this->lda = lda > 0 ? lda : rows;
    }
  }

  /**
   * @brief 访问第i行第j列的元素，兼容RowMajor和ColumnMajor
   *
   * @param i 第i行
   * @param j 第j列
   * @return T
   */
  T get(size_t i, size_t j) const {
    assert(0 <= i && i < rows);
    assert(0 <= j && j < cols);
    if constexpr (layout == StorageLayout::RowMajor) {
      return data[(start_row + i) * lda + (start_col + j)];
    } else {
      return data[(start_col + j) * lda + (start_row + i)];
    }
  }

  void set(size_t i, size_t j, T value) {
    assert(0 <= i && i < rows);
    assert(0 <= j && j < cols);
    if constexpr (layout == StorageLayout::RowMajor) {
      data[(start_row + i) * lda + (start_col + j)] = value;
    } else {
      data[(start_col + j) * lda + (start_row + i)] = value;
    }
  }

  /**
   * @brief 生成子矩阵窗口
   *
   * @param i 窗口左上角坐标 (i, j)
   * @param j 窗口左上角坐标 (i, j)
   * @param rows 窗口行数
   * @param cols 窗口列数
   * @return MatrixView
   */
  MatrixView submat(size_t i, size_t j, size_t rows, size_t cols) {
    return MatrixView<T, layout>{rows, cols, data, lda, i, j};
  }
};

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
 * @return Matrix<T> 输出矩阵
 */
template <typename T, size_t GEMM_NC = 4080, size_t GEMM_KC = 256,
          size_t GEMM_MC = 72, size_t GEMM_MR = 8, size_t GEMM_NR = 4,
          typename MICRO_KERNEL = MicroKernel<T, GEMM_MR, GEMM_NR>,
          size_t GEMM_SIMD_ALIGN_SIZE = 32>
MatrixView<T> gemm(MatrixView<T> A, MatrixView<T> B);
} // namespace GEMM