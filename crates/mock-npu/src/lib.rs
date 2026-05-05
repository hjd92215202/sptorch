use std::ffi::c_void;
use std::os::raw::c_char;
use std::slice;

static NAME: &[u8] = b"sptorch_mock_npu\0";

#[no_mangle]
pub extern "C" fn sptorch_backend_init() -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn sptorch_backend_shutdown() {}

#[no_mangle]
pub extern "C" fn sptorch_backend_name() -> *const c_char {
    NAME.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn sptorch_alloc(n: usize) -> *mut c_void {
    let mut v = Vec::<f32>::with_capacity(n);
    v.resize(n, 0.0);
    let ptr = v.as_mut_ptr();
    std::mem::forget(v);
    ptr as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // We don't know the exact capacity, but drop will handle it
    // This is a simplified mock — in production the alloc would track size
    drop(Box::from_raw(handle as *mut f32));
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_copy_h2d(host: *const f32, device: *mut c_void, n: usize) -> i32 {
    let src = slice::from_raw_parts(host, n);
    let dst = slice::from_raw_parts_mut(device as *mut f32, n);
    dst.copy_from_slice(src);
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_copy_d2h(device: *const c_void, host: *mut f32, n: usize) -> i32 {
    let src = slice::from_raw_parts(device as *const f32, n);
    let dst = slice::from_raw_parts_mut(host, n);
    dst.copy_from_slice(src);
    0
}

#[no_mangle]
pub extern "C" fn sptorch_sync() -> i32 {
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_query_runtime(queue_depth: *mut u32, online: *mut u32) -> i32 {
    if !queue_depth.is_null() {
        *queue_depth = 0;
    }
    if !online.is_null() {
        *online = 1;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_add_f32(a: *const c_void, b: *const c_void, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let b = slice::from_raw_parts(b as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        o[i] = a[i] + b[i];
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_mul_f32(a: *const c_void, b: *const c_void, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let b = slice::from_raw_parts(b as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        o[i] = a[i] * b[i];
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_neg_f32(a: *const c_void, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        o[i] = -a[i];
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_exp_f32(a: *const c_void, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        o[i] = a[i].exp();
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_log_f32(a: *const c_void, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        o[i] = a[i].ln();
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_relu_f32(a: *const c_void, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        o[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_gelu_f32(a: *const c_void, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        let x = a[i];
        o[i] = 0.5 * x * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_scale_f32(a: *const c_void, scalar: f32, out: *mut c_void, n: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, n);
    let o = slice::from_raw_parts_mut(out as *mut f32, n);
    for i in 0..n {
        o[i] = a[i] * scalar;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_matmul_f32(
    a: *const c_void,
    b: *const c_void,
    out: *mut c_void,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, m * k);
    let b = slice::from_raw_parts(b as *const f32, k * n);
    let o = slice::from_raw_parts_mut(out as *mut f32, m * n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            o[i * n + j] = sum;
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_batch_matmul_f32(
    a: *const c_void,
    b: *const c_void,
    out: *mut c_void,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> i32 {
    for bi in 0..batch {
        sptorch_matmul_f32(
            (a as *const f32).add(bi * m * k) as *const c_void,
            (b as *const f32).add(bi * k * n) as *const c_void,
            (out as *mut f32).add(bi * m * n) as *mut c_void,
            m,
            k,
            n,
        );
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn sptorch_softmax_f32(a: *const c_void, out: *mut c_void, rows: usize, cols: usize) -> i32 {
    let a = slice::from_raw_parts(a as *const f32, rows * cols);
    let o = slice::from_raw_parts_mut(out as *mut f32, rows * cols);
    for r in 0..rows {
        let row = &a[r * cols..(r + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (row[c] - max).exp();
            o[r * cols + c] = e;
            sum += e;
        }
        for c in 0..cols {
            o[r * cols + c] /= sum;
        }
    }
    0
}
