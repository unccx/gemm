#pragma once
#include <cassert>
#include <cstddef>
#include <fmt/base.h>
#include <fmt/core.h>
#include <string_view>
#include <utility>

namespace GEMM {

enum class StorageLayout : int {
  RowMajor = 0,   // 行优先
  ColumnMajor = 1 // 列优先
};

template <typename T, StorageLayout layout = StorageLayout::ColumnMajor>
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
   * @param data 指向数据的指针，
   *             需要调用者保证 data 指针所持有的内存size >= rows*cols
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

  // 重载 () 运算符，允许访问和修改元素
  T &operator()(size_t i, size_t j) {
    return data[calculateIndex(i, j)]; // 返回元素的引用，可以修改元素
  }

  // 重载 () 运算符，用于只读访问元素
  T operator()(size_t i, size_t j) const {
    return data[calculateIndex(i, j)]; // 返回元素的值
  }

  std::pair<size_t, size_t> shape() { return {rows, cols}; }

  /**
   * @brief 生成子矩阵窗口
   *
   * @param i 窗口左上角坐标 (i, j)
   * @param j 窗口左上角坐标 (i, j)
   * @param rows 窗口行数
   * @param cols 窗口列数
   * @return MatrixView
   */
  MatrixView submat(size_t i, size_t j, size_t rows, size_t cols) const {
    assert((i + rows) <= (this->rows));
    assert((j + cols) <= (this->cols));
    return MatrixView<T, layout>{rows, cols,          data,
                                 lda,  start_row + i, start_col + j};
  }

private:
  /**
   * @brief 计算第i行第j列的元素在一维内存中的索引，兼容RowMajor和ColumnMajor
   *
   * @param i 第i行
   * @param j 第j列
   * @return size_t
   */
  size_t calculateIndex(size_t i, size_t j) const {
    // if (!(0 <= i && i < rows && 0 <= j && j < cols)) {
    //   throw std::out_of_range("Index out of range");
    // }
    assert(0 <= i && i < rows);
    assert(0 <= j && j < cols);

    if constexpr (layout == StorageLayout::RowMajor) {
      return (start_row + i) * lda + (start_col + j);
    } else {
      return (start_col + j) * lda + (start_row + i);
    }
  }
};

template <typename T, StorageLayout layout>
void print_mat(const MatrixView<T, layout> mat_view, std::string_view name) {
  fmt::print("{}: ----------------------\n", name);
  for (size_t i = 0; i < mat_view.rows; i++) {
    for (size_t j = 0; j < mat_view.cols; j++) {
      fmt::print("{}\t", mat_view(i, j));
    }
    fmt::print("\n");
  }
  fmt::print("--------------------------\n");
}

// extern template声明：防止编译器实例化
extern template class MatrixView<float, StorageLayout::ColumnMajor>;
extern template class MatrixView<double, StorageLayout::ColumnMajor>;
extern template class MatrixView<float, StorageLayout::RowMajor>;
extern template class MatrixView<double, StorageLayout::RowMajor>;
} // namespace GEMM
