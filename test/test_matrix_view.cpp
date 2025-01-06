#include <GEMM/gemm.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>

SCENARIO("MatrixView 访问和修改", "[MatrixView]") {
  GIVEN("给定一个4x6的RowMajor矩阵") {
    size_t rows = 4;
    size_t cols = 6;
    size_t start_row = 0;
    size_t start_col = 0;
    float *data = new float[rows * cols]{0};

    //  0  1  2  3  4  5
    //  6  7  8  9 10 11
    // 12 13 14 15 16 17
    // 18 19 20 21 22 23

    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = i;
    }

    GEMM::MatrixView<float, GEMM::StorageLayout::RowMajor> mat_view{rows, cols,
                                                                    data};

    WHEN("访问矩阵元素") {
      THEN("lda是cols") { REQUIRE(mat_view.lda == mat_view.cols); }

      THEN("访问到的是期望值") {
        CHECK(mat_view(1, 3) == 9.0f);
        CHECK(mat_view(2, 3) == 15.0f);
        CHECK(mat_view(3, 1) == 19.0f);
        CHECK(mat_view(3, 5) == 23.0f);
      }
    }

    WHEN("修改矩阵元素") {
      mat_view(3, 1) = 101;
      mat_view(3, 5) = 102;
      mat_view(0, 0) = 100;
      THEN("修改成功") {
        CHECK(mat_view(0, 0) == 100.0f);
        CHECK(mat_view(3, 1) == 101.0f);
        CHECK(mat_view(3, 5) == 102.0f);
      }
    }

    WHEN("获取submatrix") {
      //  8  9 10
      // 14 15 16
      auto submat_view = mat_view.submat(1, 2, 2, 3);
      THEN("访问到的是期望值") {
        CHECK(submat_view(0, 0) == 8.0f);
        CHECK(submat_view(0, 1) == 9.0f);
        CHECK(submat_view(1, 1) == 15.0f);
        CHECK(submat_view(1, 2) == 16.0f);
      }
    }
  }

  GIVEN("给定一个4x6的ColumnMajor矩阵") {
    size_t rows = 4;
    size_t cols = 6;
    size_t start_row = 0;
    size_t start_col = 0;
    float *data = new float[rows * cols]{0};

    // 0  4  8  12  16  20
    // 1  5  9  13  17  21
    // 2  6  10 14  18  22
    // 3  7  11 15  19  23

    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = i;
    }

    GEMM::MatrixView<float, GEMM::StorageLayout::ColumnMajor> mat_view{
        rows, cols, data};

    WHEN("访问矩阵元素") {
      THEN("lda是rows") { REQUIRE(mat_view.lda == mat_view.rows); }

      THEN("访问到的是期望值") {
        CHECK(mat_view(1, 3) == 13.0f);
        CHECK(mat_view(2, 3) == 14.0f);
        CHECK(mat_view(3, 1) == 7.0f);
        CHECK(mat_view(3, 5) == 23.0f);
      }

      WHEN("修改矩阵元素") {
        mat_view(0, 0) = 100;
        mat_view(3, 1) = 101;
        mat_view(3, 5) = 102;
        THEN("修改成功") {
          CHECK(mat_view(0, 0) == 100.0f);
          CHECK(mat_view(3, 1) == 101.0f);
          CHECK(mat_view(3, 5) == 102.0f);
        }
      }

      WHEN("获取submatrix") {
        //  9  13  17
        // 10  14  18
        auto submat_view = mat_view.submat(1, 2, 2, 3);
        THEN("访问到的是期望值") {
          CHECK(submat_view(0, 0) == 9.0f);
          CHECK(submat_view(0, 1) == 13.0f);
          CHECK(submat_view(1, 1) == 14.0f);
          CHECK(submat_view(1, 2) == 18.0f);
        }
      }
    }
  }
}

SCENARIO("MatrixView 连续获取 submatrix") {
  GIVEN("给定一个4x6的ColumnMajor矩阵") {
    size_t rows = 4;
    size_t cols = 6;
    size_t start_row = 0;
    size_t start_col = 0;
    float *data = new float[rows * cols]{0};

    //  0  1  2  3  4  5
    //  6  7  8  9 10 11
    // 12 13 14 15 16 17
    // 18 19 20 21 22 23

    for (size_t i = 0; i < rows * cols; i++) {
      data[i] = i;
    }

    GEMM::MatrixView<float, GEMM::StorageLayout::RowMajor> mat_view{rows, cols,
                                                                    data};
    WHEN("获取submatrix的submatrix") {
      //  8  9 10
      // 14 15 16
      auto submat_view = mat_view.submat(1, 2, 2, 3);
      auto submat_view2 = submat_view.submat(1, 1, 1, 2);
      THEN("访问到的是期望值") {
        REQUIRE(submat_view2(0, 0) == 15.0f);
        REQUIRE(submat_view2(0, 1) == 16.0f);
      }
      THEN("shape符合预期") {
        auto [r, c] = submat_view.shape();
        auto [r1, c1] = submat_view2.shape();
        // fmt::print("shape:[{}, {}]\n", r, c);
        // fmt::print("shape:[{}, {}]\n", r1, c1);
        REQUIRE(r == 2);
        REQUIRE(c == 3);
        REQUIRE(r1 == 1);
        REQUIRE(c1 == 2);
      }
    }
  }
}