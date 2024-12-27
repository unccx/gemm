#include <GEMM/micro_kernel.hpp>
#include <catch2/catch_test_macros.hpp>

SCENARIO("micro_kernel的naive实现") {
  GIVEN("4x6的ColumnMajor矩阵A和6x3的RowMajor矩阵B以及AxB=C矩阵") {
    constexpr size_t m = 4;
    constexpr size_t k = 6;
    constexpr size_t n = 3;
    using T = float;
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> A{m, k,
                                                            new T[m * k]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::RowMajor> B{k, n, new T[k * n]{0}};
    GEMM::MatrixView<T, GEMM::StorageLayout::ColumnMajor> C{m, n,
                                                            new T[m * n]{0}};

    for (size_t i = 0; i < A.rows * A.cols; i++) {
      A.data[i] = i;
    }
    for (size_t i = A.rows * A.cols; i < B.rows * B.cols; i++) {
      B.data[i] = i;
    }

    for (size_t i = 0; i < C.rows; i++) {
      for (size_t j = 0; j < C.cols; j++) {
        for (size_t k = 0; k < A.cols; k++) {
          C(i, j) = A(i, j) * B(i, j);
        }
      }
    }

    WHEN("A和B矩阵相乘") {
      GEMM::MatrixView<float, GEMM::StorageLayout::ColumnMajor> output{
          m, n, new float[m * n]{0}};

      REQUIRE(A.cols == B.rows);
      GEMM::MicroKernel<float, GEMM::StorageLayout::ColumnMajor, m,
                        n>::block_update(A, B, output);

      THEN("micro_kernel结果与naive矩阵乘法相同") {
        for (size_t i = 0; i < output.rows; i++) {
          for (size_t j = 0; j < output.cols; j++) {
            REQUIRE(output(i, j) == C(i, j));
          }
        }
      }
    }
  }
}