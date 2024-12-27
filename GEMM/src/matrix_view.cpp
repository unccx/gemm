#include <GEMM/matrix_view.hpp>

namespace GEMM {
template class MatrixView<float, StorageLayout::ColumnMajor>;
template class MatrixView<double, StorageLayout::ColumnMajor>;
template class MatrixView<float, StorageLayout::RowMajor>;
template class MatrixView<double, StorageLayout::RowMajor>;
} // namespace GEMM
