/**
 * Mock NPU backend for testing the sptorch HAL FFI interface.
 * Implements all sptorch_* symbols using plain CPU (malloc + memcpy).
 * Build: gcc -shared -o mock_npu.dll mock_npu.c -lm  (Windows)
 *        gcc -shared -fPIC -o libmock_npu.so mock_npu.c -lm  (Linux)
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/sptorch_hal.h"

/* ---- lifecycle ---- */

int sptorch_backend_init(void) { return 0; }
void sptorch_backend_shutdown(void) {}
const char* sptorch_backend_name(void) { return "mock_npu"; }

/* ---- memory: just malloc, handle = float* ---- */

void* sptorch_alloc(size_t n) {
    return malloc(n * sizeof(float));
}

void sptorch_free(void* handle) {
    free(handle);
}

int sptorch_copy_h2d(const float* host, void* device, size_t n) {
    memcpy(device, host, n * sizeof(float));
    return 0;
}

int sptorch_copy_d2h(const void* device, float* host, size_t n) {
    memcpy(host, device, n * sizeof(float));
    return 0;
}

int sptorch_sync(void) { return 0; }

/* ---- element-wise ops ---- */

int sptorch_add_f32(const void* a, const void* b, void* out, size_t n) {
    const float* fa = (const float*)a;
    const float* fb = (const float*)b;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) fo[i] = fa[i] + fb[i];
    return 0;
}

int sptorch_mul_f32(const void* a, const void* b, void* out, size_t n) {
    const float* fa = (const float*)a;
    const float* fb = (const float*)b;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) fo[i] = fa[i] * fb[i];
    return 0;
}

int sptorch_neg_f32(const void* a, void* out, size_t n) {
    const float* fa = (const float*)a;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) fo[i] = -fa[i];
    return 0;
}

int sptorch_exp_f32(const void* a, void* out, size_t n) {
    const float* fa = (const float*)a;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) fo[i] = expf(fa[i]);
    return 0;
}

int sptorch_log_f32(const void* a, void* out, size_t n) {
    const float* fa = (const float*)a;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) fo[i] = logf(fa[i]);
    return 0;
}

int sptorch_relu_f32(const void* a, void* out, size_t n) {
    const float* fa = (const float*)a;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) fo[i] = fa[i] > 0.0f ? fa[i] : 0.0f;
    return 0;
}

int sptorch_gelu_f32(const void* a, void* out, size_t n) {
    const float* fa = (const float*)a;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) {
        float x = fa[i];
        fo[i] = 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / 3.14159265f) * (x + 0.044715f * x * x * x)));
    }
    return 0;
}

int sptorch_scale_f32(const void* a, float scalar, void* out, size_t n) {
    const float* fa = (const float*)a;
    float* fo = (float*)out;
    for (size_t i = 0; i < n; i++) fo[i] = fa[i] * scalar;
    return 0;
}

/* ---- matmul ---- */

int sptorch_matmul_f32(const void* a, const void* b, void* out,
                       size_t m, size_t k, size_t n) {
    const float* fa = (const float*)a;
    const float* fb = (const float*)b;
    float* fo = (float*)out;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += fa[i * k + p] * fb[p * n + j];
            }
            fo[i * n + j] = sum;
        }
    }
    return 0;
}

int sptorch_batch_matmul_f32(const void* a, const void* b, void* out,
                             size_t batch, size_t m, size_t k, size_t n) {
    size_t a_stride = m * k;
    size_t b_stride = k * n;
    size_t o_stride = m * n;
    for (size_t bi = 0; bi < batch; bi++) {
        sptorch_matmul_f32(
            (const float*)a + bi * a_stride,
            (const float*)b + bi * b_stride,
            (float*)out + bi * o_stride,
            m, k, n);
    }
    return 0;
}

/* ---- reduction ---- */

int sptorch_softmax_f32(const void* a, void* out, size_t rows, size_t cols) {
    const float* fa = (const float*)a;
    float* fo = (float*)out;
    for (size_t r = 0; r < rows; r++) {
        const float* row = fa + r * cols;
        float* orow = fo + r * cols;
        float max_val = row[0];
        for (size_t c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            orow[c] = expf(row[c] - max_val);
            sum += orow[c];
        }
        for (size_t c = 0; c < cols; c++) {
            orow[c] /= sum;
        }
    }
    return 0;
}
