/**
 * sptorch Hardware Abstraction Layer - External Backend C API
 *
 * Third-party hardware vendors implement these functions in a shared library
 * (.so / .dll / .dylib). sptorch loads the library at runtime via hal-ffi.
 *
 * All arrays are row-major, f32. The vendor owns device memory; sptorch
 * only sees opaque handles (void*).
 */

#ifndef SPTORCH_HAL_FFI_H
#define SPTORCH_HAL_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- lifecycle ---- */

/** Initialize the backend. Returns 0 on success. */
int sptorch_backend_init(void);

/** Shutdown and release all resources. */
void sptorch_backend_shutdown(void);

/** Human-readable backend name, e.g. "tank9k". */
const char* sptorch_backend_name(void);

/* ---- memory management ---- */

/** Allocate device memory for `n` floats. Returns opaque handle, NULL on failure. */
void* sptorch_alloc(size_t n);

/** Free device memory. */
void sptorch_free(void* handle);

/** Copy `n` floats from host to device. */
int sptorch_copy_h2d(const float* host, void* device, size_t n);

/** Copy `n` floats from device to host. */
int sptorch_copy_d2h(const void* device, float* host, size_t n);

/** Synchronize (wait for all pending ops). */
int sptorch_sync(void);

/* ---- element-wise ops ---- */

int sptorch_add_f32(const void* a, const void* b, void* out, size_t n);
int sptorch_mul_f32(const void* a, const void* b, void* out, size_t n);
int sptorch_neg_f32(const void* a, void* out, size_t n);
int sptorch_exp_f32(const void* a, void* out, size_t n);
int sptorch_log_f32(const void* a, void* out, size_t n);
int sptorch_relu_f32(const void* a, void* out, size_t n);
int sptorch_gelu_f32(const void* a, void* out, size_t n);
int sptorch_scale_f32(const void* a, float scalar, void* out, size_t n);

/* ---- matmul ---- */

/** C = A @ B, row-major. A:[m,k] B:[k,n] C:[m,n] */
int sptorch_matmul_f32(const void* a, const void* b, void* out,
                       size_t m, size_t k, size_t n);

/** Batched matmul. A:[batch,m,k] B:[batch,k,n] C:[batch,m,n] */
int sptorch_batch_matmul_f32(const void* a, const void* b, void* out,
                             size_t batch, size_t m, size_t k, size_t n);

/* ---- reduction ---- */

/** Row-wise softmax. input/output: [rows, cols] */
int sptorch_softmax_f32(const void* a, void* out, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif /* SPTORCH_HAL_FFI_H */
