#include <GEMM/gemm.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <fmt/base.h>
#include <fmt/core.h>

template <typename T, GEMM::StorageLayout layout>
void print_mat(const GEMM::MatrixView<T, layout> mat_view) {
  fmt::print("mat_view: \n");
  for (size_t i = 0; i < mat_view.rows; i++) {
    for (size_t j = 0; j < mat_view.cols; j++) {
      fmt::print("{} ", mat_view(i, j));
    }
    fmt::print("\n");
  }
}

SCENARIO("pack_panel 处理一个 panel") {
  GIVEN("给定一个20x30的ColumnMajor矩阵") {
    size_t rows = 5;
    size_t cols = 6;
    size_t start_row = 0;
    size_t start_col = 0;
    float *data = new float[rows * cols]{0};

    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = i;
    }

    GEMM::MatrixView<float, GEMM::StorageLayout::ColumnMajor> mat_view{
        rows, cols, data};

    // print_mat(mat_view);
    // 0 5 10 15 20 25
    // 1 6 11 16 21 26
    // 2 7 12 17 22 27
    // 3 8 13 18 23 28
    // 4 9 14 19 24 29

    WHEN("对子矩阵进行RowMajor的data reorder") {
      float *pack_ptr = (float *)new float[rows * cols]{0};

      auto sub_matrix = mat_view.submat(1, 1, 3, 4);
      GEMM::MatrixView<float, GEMM::StorageLayout::RowMajor> pack =
          GEMM::pack_panel<float, GEMM::StorageLayout::ColumnMajor, 4,
                           GEMM::StorageLayout::RowMajor>(sub_matrix, pack_ptr);

      // print_mat(sub_matrix);
      // print_mat(pack);

      // fmt::print("pack:\n");
      // for (size_t i = 0; i < pack.rows * pack.cols; i++) {
      //   fmt::print("{} ", pack.data[i]);
      // }
      // fmt::print("\n");

      THEN("data reorder 前后数据通过索引访问结果一致") {
        REQUIRE(sub_matrix.rows == pack.rows);
        REQUIRE(sub_matrix.cols == pack.cols);

        for (size_t i = 0; i < sub_matrix.rows; i++) {
          for (size_t j = 0; j < sub_matrix.cols; j++) {
            REQUIRE(sub_matrix(i, j) == pack(i, j));
          }
        }
      }
    }

    WHEN("对子矩阵进行RowMajor的data reorder, packed_panel需要padding") {
      float *pack_ptr = (float *)new float[rows * cols]{0};

      auto sub_matrix = mat_view.submat(1, 4, 3, 2);
      GEMM::MatrixView<float, GEMM::StorageLayout::RowMajor> pack =
          GEMM::pack_panel<float, GEMM::StorageLayout::ColumnMajor, 4,
                           GEMM::StorageLayout::RowMajor>(sub_matrix, pack_ptr);

      // print_mat(sub_matrix);
      // print_mat(pack);

      // fmt::print("pack:\n");
      // for (size_t i = 0; i < pack.rows * pack.cols; i++) {
      //   fmt::print("{} ", pack.data[i]);
      // }
      // fmt::print("\n");

      THEN("data reorder 前后数据通过索引访问结果一致") {
        REQUIRE(sub_matrix.rows <= pack.rows);
        REQUIRE(sub_matrix.cols <= pack.cols);

        for (size_t i = 0; i < sub_matrix.rows; i++) {
          for (size_t j = 0; j < sub_matrix.cols; j++) {
            REQUIRE(sub_matrix(i, j) == pack(i, j));
          }
        }
      }
    }

    WHEN("对原矩阵进行data reorder") {
      float *pack_ptr = (float *)new float[rows * (cols + 2)]{0};
      constexpr size_t GEMM_NR = 4;
      size_t k_c = rows;
      size_t n_c = cols;
      THEN("") {
        // pack into B_panel
        for (size_t j = 0; j < n_c; j += GEMM_NR) { // j 是 B_panel 中的列索引
          size_t n_r = std::min(n_c - j, GEMM_NR);
          auto sub_matrix = mat_view.submat(0, j, k_c, n_r);
          auto pack = GEMM::pack_panel<float, GEMM::StorageLayout::ColumnMajor,
                                       GEMM_NR, GEMM::StorageLayout::RowMajor>(
              sub_matrix, &pack_ptr[j * k_c]);
          for (size_t i = 0; i < sub_matrix.rows; i++) {
            for (size_t k = 0; k < sub_matrix.cols; k++) {
              REQUIRE(sub_matrix(i, k) == pack(i, k));
            }
          }
        }
      }
    }
  }
}

SCENARIO("gemm") {
  GIVEN("float类型ColumnMajor矩阵A和ColumnMajors矩阵B以及AxB=C矩阵") {
    constexpr size_t m = 1024;
    constexpr size_t k = 1000;
    constexpr size_t n = 400;
    using T = float;
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> A{m, k,
                                                            new T[m * k]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> B{k, n,
                                                            new T[k * n]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> C{m, n,
                                                            new T[m * n]{0}};

    // A B C 填充数据
    for (size_t i = 0; i < A.rows * A.cols; i++) {
      A.data[i] = i;
    }
    for (size_t i = A.rows * A.cols; i < B.rows * B.cols; i++) {
      B.data[i] = i;
    }
    // AxB=C naive实现
    for (size_t i = 0; i < C.rows; i++) {
      for (size_t j = 0; j < C.cols; j++) {
        for (size_t k = 0; k < A.cols; k++) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }

    WHEN("使用gemm函数AxB") {
      REQUIRE(A.cols == B.rows);
      auto output = GEMM::gemm(A, B);
      THEN("gemm结果与naive矩阵乘法相同") {
        for (size_t i = 0; i < output.rows; i++) {
          for (size_t j = 0; j < output.cols; j++) {
            REQUIRE(output(i, j) == C(i, j));
          }
        }
      }
    }
  }

  GIVEN("double类型ColumnMajor矩阵A和ColumnMajors矩阵B以及AxB=C矩阵") {
    constexpr size_t m = 1024;
    constexpr size_t k = 1000;
    constexpr size_t n = 400;
    using T = double;
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> A{m, k,
                                                            new T[m * k]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> B{k, n,
                                                            new T[k * n]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> C{m, n,
                                                            new T[m * n]{0}};

    // A B C 填充数据
    for (size_t i = 0; i < A.rows * A.cols; i++) {
      A.data[i] = i;
    }
    for (size_t i = A.rows * A.cols; i < B.rows * B.cols; i++) {
      B.data[i] = i;
    }
    // AxB=C naive实现
    for (size_t i = 0; i < C.rows; i++) {
      for (size_t j = 0; j < C.cols; j++) {
        for (size_t k = 0; k < A.cols; k++) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }

    WHEN("使用gemm函数AxB") {
      REQUIRE(A.cols == B.rows);
      auto output = GEMM::gemm(A, B);
      THEN("gemm结果与naive矩阵乘法相同") {
        for (size_t i = 0; i < output.rows; i++) {
          for (size_t j = 0; j < output.cols; j++) {
            REQUIRE(output(i, j) == C(i, j));
          }
        }
      }
    }
  }

  GIVEN("int8_t类型ColumnMajor矩阵A和ColumnMajors矩阵B以及AxB=C矩阵") {
    constexpr size_t m = 1024;
    constexpr size_t k = 1000;
    constexpr size_t n = 400;
    using T = int8_t;
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> A{m, k,
                                                            new T[m * k]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> B{k, n,
                                                            new T[k * n]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> C{m, n,
                                                            new T[m * n]{0}};

    // A B C 填充数据
    for (size_t i = 0; i < A.rows * A.cols; i++) {
      A.data[i] = i;
    }
    for (size_t i = A.rows * A.cols; i < B.rows * B.cols; i++) {
      B.data[i] = i;
    }
    // AxB=C naive实现
    for (size_t i = 0; i < C.rows; i++) {
      for (size_t j = 0; j < C.cols; j++) {
        for (size_t k = 0; k < A.cols; k++) {
          C(i, j) += A(i, k) * B(k, j);
        }
      }
    }

    WHEN("使用gemm函数AxB") {
      REQUIRE(A.cols == B.rows);
      auto output = GEMM::gemm(A, B);
      THEN("gemm结果与naive矩阵乘法相同") {
        for (size_t i = 0; i < output.rows; i++) {
          for (size_t j = 0; j < output.cols; j++) {
            REQUIRE(output(i, j) == C(i, j));
          }
        }
      }
    }
  }
}
