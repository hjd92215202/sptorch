use cudarc::driver::*;
use cudarc::cublas::{CudaBlas, sys::cublasOperation_t};
use std::sync::Arc;

#[derive(Debug)]
pub enum CudaError {
    Driver(DriverError),
    Cublas(String),
    Compile(String),
}

impl From<DriverError> for CudaError {
    fn from(e: DriverError) -> Self { CudaError::Driver(e) }
}

pub struct CudaBackend {
    pub dev: Arc<CudaDevice>,
    pub blas: Arc<CudaBlas>,
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let dev = CudaDevice::new(ordinal)?;
        let blas = Arc::new(CudaBlas::new(dev.clone()).map_err(|e| CudaError::Cublas(e.to_string()))?);
        Ok(CudaBackend { dev, blas })
    }
}

// ============ GPU Buffer wrapper ============

pub struct GpuTensor {
    pub data: CudaSlice<f32>,
    pub shape: Vec<usize>,
    pub numel: usize,
}

impl GpuTensor {
    pub fn from_host(backend: &CudaBackend, data: &[f32], shape: Vec<usize>) -> Result<Self, CudaError> {
        let numel = data.len();
        let gpu_data = backend.dev.htod_sync_copy(data)?;
        Ok(GpuTensor { data: gpu_data, shape, numel })
    }

    pub fn to_host(&self, backend: &CudaBackend) -> Result<Vec<f32>, CudaError> {
        let host = backend.dev.dtoh_sync_copy(&self.data)?;
        Ok(host)
    }

    pub fn zeros(backend: &CudaBackend, shape: Vec<usize>) -> Result<Self, CudaError> {
        let numel: usize = shape.iter().product();
        let gpu_data = backend.dev.alloc_zeros::<f32>(numel)?;
        Ok(GpuTensor { data: gpu_data, shape, numel })
    }
}

// ============ CUDA C Kernels ============

const KERNEL_SRC: &str = r#"
extern "C" __global__ void add_kernel(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

extern "C" __global__ void mul_kernel(const float* a, const float* b, float* c, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

extern "C" __global__ void neg_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = -a[idx];
}

extern "C" __global__ void scale_kernel(const float* a, float* b, float scalar, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = a[idx] * scalar;
}

extern "C" __global__ void exp_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = expf(a[idx]);
}

extern "C" __global__ void log_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = logf(a[idx]);
}

extern "C" __global__ void gelu_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = a[idx];
        float k = 0.7978845608f * (x + 0.044715f * x * x * x);
        b[idx] = 0.5f * x * (1.0f + tanhf(k));
    }
}

extern "C" __global__ void relu_kernel(const float* a, float* b, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
}
"#;

const KERNEL_NAMES: &[&str] = &[
    "add_kernel", "mul_kernel", "neg_kernel", "scale_kernel",
    "exp_kernel", "log_kernel", "gelu_kernel", "relu_kernel",
];

// ============ Kernel Operations ============

fn launch_cfg(n: usize) -> LaunchConfig {
    let block = 256;
    let grid = (n + block - 1) / block;
    LaunchConfig {
        grid_dim: (grid as u32, 1, 1),
        block_dim: (block as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaBackend {
    pub fn load_kernels(&self) -> Result<(), CudaError> {
        if self.dev.has_func("elem", "add_kernel") {
            return Ok(());
        }
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC)
            .map_err(|e| CudaError::Compile(e.to_string()))?;
        self.dev.load_ptx(ptx, "elem", KERNEL_NAMES)?;
        Ok(())
    }

    pub fn gpu_add(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        assert_eq!(a.numel, b.numel);
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "add_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &b.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }

    pub fn gpu_mul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        assert_eq!(a.numel, b.numel);
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "mul_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &b.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }

    pub fn gpu_neg(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "neg_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }

    pub fn gpu_scale(&self, a: &GpuTensor, scalar: f32) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "scale_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, scalar, a.numel as u32)) }?;
        Ok(out)
    }

    pub fn gpu_exp(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "exp_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }

    pub fn gpu_log(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "log_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }

    pub fn gpu_gelu(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "gelu_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }

    pub fn gpu_relu(&self, a: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let mut out = GpuTensor::zeros(self, a.shape.clone())?;
        let f = self.dev.get_func("elem", "relu_kernel").unwrap();
        let cfg = launch_cfg(a.numel);
        unsafe { f.launch(cfg, (&a.data, &mut out.data, a.numel as u32)) }?;
        Ok(out)
    }

    /// matmul via cuBLAS: C = A @ B, A:[m,k], B:[k,n] -> C:[m,n]
    pub fn gpu_matmul(&self, a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, CudaError> {
        let (m, k) = (a.shape[0], a.shape[1]);
        let n = b.shape[1];
        assert_eq!(a.shape[1], b.shape[0]);

        let mut out = GpuTensor::zeros(self, vec![m, n])?;

        // cuBLAS is column-major. For row-major A@B:
        // treat as col-major B^T @ A^T = (A@B)^T, read back as row-major = A@B
        unsafe {
            cudarc::cublas::result::sgemm(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &1.0f32 as *const f32,
                *b.data.device_ptr() as *const f32,
                n as i32,
                *a.data.device_ptr() as *const f32,
                k as i32,
                &0.0f32 as *const f32,
                *out.data.device_ptr_mut() as *mut f32,
                n as i32,
            )
        }.map_err(|e| CudaError::Cublas(e.to_string()))?;

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> CudaBackend {
        let backend = CudaBackend::new(0).expect("Failed to create CUDA backend");
        backend.load_kernels().expect("Failed to load kernels");
        backend
    }

    #[test]
    fn test_host_roundtrip() {
        let b = setup();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gpu = GpuTensor::from_host(&b, &data, vec![5]).unwrap();
        let host = gpu.to_host(&b).unwrap();
        assert_eq!(host, data);
    }

    #[test]
    fn test_gpu_add() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let c = GpuTensor::from_host(&b, &[5.0, 6.0, 7.0, 8.0], vec![4]).unwrap();
        let out = b.gpu_add(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_gpu_mul() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[2.0, 3.0, 4.0, 5.0], vec![4]).unwrap();
        let c = GpuTensor::from_host(&b, &[3.0, 4.0, 5.0, 6.0], vec![4]).unwrap();
        let out = b.gpu_mul(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_gpu_neg() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, -2.0, 3.0], vec![3]).unwrap();
        let out = b.gpu_neg(&a).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_gpu_scale() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0], vec![3]).unwrap();
        let out = b.gpu_scale(&a, 2.5).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![2.5, 5.0, 7.5]);
    }

    #[test]
    fn test_gpu_exp() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[0.0, 1.0], vec![2]).unwrap();
        let out = b.gpu_exp(&a).unwrap();
        let r = out.to_host(&b).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_log() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, std::f32::consts::E], vec![2]).unwrap();
        let out = b.gpu_log(&a).unwrap();
        let r = out.to_host(&b).unwrap();
        assert!((r[0] - 0.0).abs() < 1e-6);
        assert!((r[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gpu_gelu() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
        let out = b.gpu_gelu(&a).unwrap();
        let r = out.to_host(&b).unwrap();
        assert!((r[1] - 0.0).abs() < 1e-6); // gelu(0) = 0
        assert!(r[0] < 0.0);                  // gelu(-1) < 0
        assert!(r[2] > 0.8);                  // gelu(1) ~ 0.841
    }

    #[test]
    fn test_gpu_relu() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
        let out = b.gpu_relu(&a).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gpu_matmul() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let c = GpuTensor::from_host(&b, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let out = b.gpu_matmul(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gpu_matmul_square() {
        let b = setup();
        let a = GpuTensor::from_host(&b, &[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = GpuTensor::from_host(&b, &[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let out = b.gpu_matmul(&a, &c).unwrap();
        assert_eq!(out.to_host(&b).unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gpu_vs_cpu_matmul_large() {
        let b = setup();
        let m = 64;
        let k = 48;
        let n = 32;
        let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32) * 0.01).collect();

        let ga = GpuTensor::from_host(&b, &a_data, vec![m, k]).unwrap();
        let gb = GpuTensor::from_host(&b, &b_data, vec![k, n]).unwrap();
        let gpu_out = b.gpu_matmul(&ga, &gb).unwrap().to_host(&b).unwrap();

        // CPU reference
        let mut cpu_out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    cpu_out[i * n + j] += a_data[i * k + p] * b_data[p * n + j];
                }
            }
        }

        for i in 0..m*n {
            assert!((gpu_out[i] - cpu_out[i]).abs() < 0.1,
                "mismatch at {}: gpu={} cpu={}", i, gpu_out[i], cpu_out[i]);
        }
    }
}
