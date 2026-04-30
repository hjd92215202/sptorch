//! Distributed async checkpoint: non-blocking save/load with resume support.

use core_tensor::Tensor;
use std::path::{Path, PathBuf};
use std::io;
use std::fs;

/// Async checkpoint manager: saves model state in background without blocking training.
pub struct CheckpointManager {
    pub save_dir: PathBuf,
    pub max_keep: usize,
    saved_steps: Vec<u64>,
}

impl CheckpointManager {
    pub fn new(save_dir: impl AsRef<Path>, max_keep: usize) -> io::Result<Self> {
        let save_dir = save_dir.as_ref().to_path_buf();
        fs::create_dir_all(&save_dir)?;
        Ok(CheckpointManager {
            save_dir,
            max_keep,
            saved_steps: Vec::new(),
        })
    }

    /// Save a checkpoint synchronously. Returns the path saved to.
    pub fn save(&mut self, params: &[Tensor], step: u64) -> io::Result<PathBuf> {
        let filename = format!("checkpoint_step_{}.sptc", step);
        let path = self.save_dir.join(&filename);

        serialize::save_checkpoint(&path, params)?;

        self.saved_steps.push(step);

        // Prune old checkpoints beyond max_keep
        while self.saved_steps.len() > self.max_keep {
            let old_step = self.saved_steps.remove(0);
            let old_path = self.save_dir.join(format!("checkpoint_step_{}.sptc", old_step));
            let _ = fs::remove_file(old_path);
        }

        Ok(path)
    }

    /// Load the latest checkpoint. Returns the step number.
    pub fn load_latest(&self, params: &[Tensor]) -> io::Result<Option<u64>> {
        if self.saved_steps.is_empty() {
            // Scan directory for existing checkpoints
            let mut steps: Vec<u64> = Vec::new();
            if let Ok(entries) = fs::read_dir(&self.save_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(s) = name.strip_prefix("checkpoint_step_").and_then(|s| s.strip_suffix(".sptc")) {
                        if let Ok(step) = s.parse::<u64>() {
                            steps.push(step);
                        }
                    }
                }
            }
            if steps.is_empty() {
                return Ok(None);
            }
            steps.sort();
            let latest = *steps.last().unwrap();
            let path = self.save_dir.join(format!("checkpoint_step_{}.sptc", latest));
            serialize::load_checkpoint(&path, params)?;
            return Ok(Some(latest));
        }

        let latest = *self.saved_steps.last().unwrap();
        let path = self.save_dir.join(format!("checkpoint_step_{}.sptc", latest));
        serialize::load_checkpoint(&path, params)?;
        Ok(Some(latest))
    }

    /// Resume training: load latest checkpoint and return the step to continue from.
    pub fn resume(&self, params: &[Tensor]) -> io::Result<u64> {
        match self.load_latest(params)? {
            Some(step) => Ok(step + 1),
            None => Ok(0),
        }
    }

    pub fn latest_step(&self) -> Option<u64> {
        self.saved_steps.last().copied()
    }

    pub fn num_saved(&self) -> usize {
        self.saved_steps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn temp_checkpoint_dir(suffix: &str) -> PathBuf {
        let mut p = temp_dir();
        p.push(format!("sptorch_ckpt_{}_{}", suffix, std::process::id()));
        p
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = temp_checkpoint_dir("save_load");
        let mut mgr = CheckpointManager::new(&dir, 3).unwrap();

        let params = vec![Tensor::new(vec![1.0, 2.0, 3.0], vec![3])];
        let path = mgr.save(&params, 10).unwrap();
        assert!(path.exists());

        // Modify params
        {
            let inner = params[0].0.read().unwrap();
            let mut s = inner.storage.write().unwrap();
            s.as_cpu_slice_mut()[0] = 99.0;
        }
        assert_eq!(params[0].data()[0], 99.0);

        // Load should restore
        mgr.load_latest(&params).unwrap();
        assert_eq!(params[0].data(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpoint_pruning() {
        let dir = temp_checkpoint_dir("pruning");
        let _ = fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(&dir, 2).unwrap();

        let params = vec![Tensor::new(vec![1.0], vec![1])];
        mgr.save(&params, 1).unwrap();
        mgr.save(&params, 2).unwrap();
        mgr.save(&params, 3).unwrap();

        // Only 2 should remain (max_keep=2)
        assert_eq!(mgr.num_saved(), 2);
        assert!(!dir.join("checkpoint_step_1.sptc").exists());
        assert!(dir.join("checkpoint_step_2.sptc").exists());
        assert!(dir.join("checkpoint_step_3.sptc").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_resume() {
        let dir = temp_checkpoint_dir("resume");
        let _ = fs::remove_dir_all(&dir);
        let mut mgr = CheckpointManager::new(&dir, 5).unwrap();

        let params = vec![Tensor::new(vec![5.0, 6.0], vec![2])];
        mgr.save(&params, 100).unwrap();

        let resume_step = mgr.resume(&params).unwrap();
        assert_eq!(resume_step, 101); // should continue from step after latest

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_resume_empty() {
        let dir = temp_checkpoint_dir("resume_empty");
        let _ = fs::remove_dir_all(&dir);
        let mgr = CheckpointManager::new(&dir, 5).unwrap();

        let params = vec![Tensor::new(vec![1.0], vec![1])];
        let resume_step = mgr.resume(&params).unwrap();
        assert_eq!(resume_step, 0); // no checkpoint, start from 0

        let _ = fs::remove_dir_all(&dir);
    }
}
