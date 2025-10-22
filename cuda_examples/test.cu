#include "include/vector_add.cuh"
#include <iostream>

void test_vector_add() {
  const int N = 10;
  float a[N], b[N], c[N];
  float a_ref[N], b_ref[N], c_ref[N];

  // 初始化输入
  for (int i = 0; i < N; ++i) {
    a[i] = i * 1.0f;
    b[i] = (N - i) * 1.0f;
    a_ref[i] = a[i];
    b_ref[i] = b[i];
  }

  // 调用向量加法
  vector_add(a, b, c, N);
  vector_add_cpu(a_ref, b_ref, c_ref, N);

  // 验证结果
  for (int i = 0; i < N; ++i) {
    if (fabs(c[i] - c_ref[i]) > 1e-5) {
      std::cerr << "Mismatch at index " << i << ": " << c[i]
                << " != " << c_ref[i] << std::endl;
      return;
    }
  }
  std::cout << "Test passed!" << std::endl;
}

int main() {
  test_vector_add();
  return 0;
}
