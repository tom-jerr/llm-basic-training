#pragma once

#include <cuda_runtime.h>

// 主机端封装函数（供 test.cu 调用）
void vector_add(const float *a, const float *b, float *c, int n);
void vector_add_cpu(const float *a, const float *b, float *c, int n);