#pragma once
#include <cstdio>
#include <immintrin.h>
namespace GEMM {
/**
 * @brief 打印_m256 (fp32 * 8)的值
 *
 * @param vec
 */
static void print_m256(__m256 vec) {
  // 创建一个浮点数组用于存储 `__m256` 的值
  float values[8];
  // 使用 `_mm256_storeu_ps` 将 `__m256` 的值存储到数组中
  _mm256_storeu_ps(values, vec);

  // 打印数组的每个元素
  std::printf("Values in __m256: ");
  for (int i = 0; i < 8; ++i) {
    std::printf("%f ", values[i]);
  }
  std::printf("\n");
}

/**
 * @brief 把一个字节的 mmask8转换成 __m256 类型的 mmask256
 *
 * @param mmask8 被转换的 mask
 * @return __m256
 */
static __m256 mmask8_to_mmask256(char mmask8) {
  float mask_arr[8];
  for (int i = 0; i < 8; ++i) {
    mask_arr[i] = mmask8 & (1 << i) ? -1.0f : 0.0f;
  }
  return _mm256_loadu_ps(mask_arr);
}
} // namespace GEMM