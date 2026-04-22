use core_tensor::DType;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub backend: String,
    pub ordinal: usize,
}

impl DeviceId {
    pub fn cpu() -> Self {
        DeviceId { backend: "cpu".into(), ordinal: 0 }
    }

    pub fn cuda(ordinal: usize) -> Self {
        DeviceId { backend: "cuda".into(), ordinal }
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend, self.ordinal)
    }
}

pub struct RawBuffer {
    pub data: Vec<u8>,
    pub device: DeviceId,
}

impl fmt::Debug for RawBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawBuffer")
            .field("len", &self.data.len())
            .field("device", &self.device)
            .finish()
    }
}

#[derive(Debug)]
pub enum HalError {
    DeviceNotFound(DeviceId),
    AllocationFailed { size: usize, reason: String },
    DeviceMismatch { expected: DeviceId, got: DeviceId },
    DTypeMismatch { expected: DType, got: DType },
    Unsupported(String),
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HalError::DeviceNotFound(id) => write!(f, "device not found: {}", id),
            HalError::AllocationFailed { size, reason } => {
                write!(f, "allocation of {} bytes failed: {}", size, reason)
            }
            HalError::DeviceMismatch { expected, got } => {
                write!(f, "device mismatch: expected {}, got {}", expected, got)
            }
            HalError::DTypeMismatch { expected, got } => {
                write!(f, "dtype mismatch: expected {:?}, got {:?}", expected, got)
            }
            HalError::Unsupported(msg) => write!(f, "unsupported: {}", msg),
        }
    }
}

impl std::error::Error for HalError {}

pub type HalResult<T> = Result<T, HalError>;

pub trait Backend: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn device_id(&self) -> DeviceId;
    fn allocate(&self, size: usize) -> HalResult<RawBuffer>;
    fn copy_to_host(&self, buf: &RawBuffer, dst: &mut [u8]) -> HalResult<()>;
    fn copy_from_host(&self, src: &[u8], buf: &mut RawBuffer) -> HalResult<()>;
    fn synchronize(&self) -> HalResult<()>;
}

pub trait KernelProvider: Backend {
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);
}

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &str { "cpu" }

    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn allocate(&self, size: usize) -> HalResult<RawBuffer> {
        Ok(RawBuffer {
            data: vec![0u8; size],
            device: DeviceId::cpu(),
        })
    }

    fn copy_to_host(&self, buf: &RawBuffer, dst: &mut [u8]) -> HalResult<()> {
        dst.copy_from_slice(&buf.data);
        Ok(())
    }

    fn copy_from_host(&self, src: &[u8], buf: &mut RawBuffer) -> HalResult<()> {
        buf.data.copy_from_slice(src);
        Ok(())
    }

    fn synchronize(&self) -> HalResult<()> { Ok(()) }
}

impl KernelProvider for CpuBackend {
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] + b[i];
        }
    }

    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        for i in 0..out.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_add() {
        let backend = CpuBackend;
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut out = vec![0.0f32; 3];
        backend.add_f32(&a, &b, &mut out);
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cpu_backend_mul() {
        let backend = CpuBackend;
        let a = vec![2.0f32, 3.0];
        let b = vec![4.0f32, 5.0];
        let mut out = vec![0.0f32; 2];
        backend.mul_f32(&a, &b, &mut out);
        assert_eq!(out, vec![8.0, 15.0]);
    }

    #[test]
    fn test_cpu_backend_matmul() {
        let backend = CpuBackend;
        // [2,3] @ [3,2] = [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut out = vec![0.0f32; 4];
        backend.matmul_f32(&a, &b, &mut out, 2, 3, 2);
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_cpu_backend_allocate() {
        let backend = CpuBackend;
        let buf = backend.allocate(16).unwrap();
        assert_eq!(buf.data.len(), 16);
        assert_eq!(buf.device, DeviceId::cpu());
    }
}
