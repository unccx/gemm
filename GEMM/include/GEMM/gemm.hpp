#pragma once
#include <GEMM/micro_kernel.hpp>
#include <algorithm>
#include <cstddef>
#include <utility>

namespace GEMM {
enum class StorageLayout {
  RowMajor = 0,   // 行优先
  ColumnMajor = 1 // 列优先
};

template <typename T> class Matrix {
private:
  size_t rows, cols;
  size_t lda;
  T *data;
  StorageLayout layout;

public:
  // ctor
  Matrix(size_t rows, size_t cols,
         StorageLayout layout = StorageLayout::RowMajor)
      : rows(rows), cols(cols), layout(layout),
        data(rows * cols ? new T[rows * cols]{0} : nullptr) {}

  // copy ctor
  Matrix(const Matrix &other)
      : rows(other.rows), cols(other.cols), layout(other.layout),
        data(rows * cols ? new T[rows * cols]{0} : nullptr) {
    std::copy(other.data, other.data + rows * cols, data);
  }

  // move ctor
  Matrix(Matrix &&other) : Matrix(0, 0) { swap(*this, other); }

  friend void swap(Matrix &left, Matrix &right) {
    using std::swap;
    swap(left.rows, right.rows);
    swap(left.cols, right.cols);
    swap(left.data, right.data);
    swap(left.layout, right.layout);
  }

  T get(size_t i, size_t j) {
    if (layout == StorageLayout::RowMajor) {
      return data[i * cols + j]; // 行优先索引
    } else {
      return data[j * rows + i]; // 列优先索引
    }
  }

  void set(size_t i, size_t j, T value) {
    if (layout == StorageLayout::RowMajor) {
      data[i * cols + j] = value; // 行优先索引
    } else {
      data[j * rows + i] = value; // 列优先索引
    }
  }

  ~Matrix() { delete[] data; }
};

/**
 * @brief 通用矩阵-矩阵乘法 A{m,k} x B{k,n} = C{m,n} ,矩阵元素为 T
 *
 * @tparam T 矩阵元素类型
 * @tparam MICRO_KERNEL
 * @param m
 * @param k
 * @param n
 * @param A   输入矩阵
 * @param lda A矩阵的leading dimension表示A[i][j]和A[i+1][j]在一维内存布局中相差
 *            lda 个元素 A[i][j] = A + lda * i + j
 * @param B   输入矩阵
 * @param ldb B矩阵的leading dimension
 * @param C   输出矩阵
 * @param ldc C矩阵的leading dimension
 */
template <typename T, size_t GEMM_KC = 256, size_t GEMM_NC = 4080,
          size_t GEMM_MC = 72, size_t GEMM_MR = 8, size_t GEMM_NR = 4,
          typename MICRO_KERNEL = MicroKernel<T, GEMM_MR, GEMM_NR>,
          size_t GEMM_SIMD_ALIGN_SIZE = 32>
void gemm(size_t m, size_t k, size_t n, T *A, size_t lda, T *B, size_t ldb,
          T *C, size_t ldc);

extern template void gemm<float>(std::size_t m, std::size_t k, std::size_t n,
                                 float *A, std::size_t lda, float *B,
                                 std::size_t ldb, float *C, std::size_t ldc);
extern template void gemm<double>(std::size_t m, std::size_t k, std::size_t n,
                                  double *A, std::size_t lda, double *B,
                                  std::size_t ldb, double *C, std::size_t ldc);
} // namespace GEMM