//! Public facade crate for SPTorch framework.
//!
//! External products should prefer `sptorch::v1::*` as the stable API surface.

pub mod v1 {
    /// Stable neural API used by product runtimes.
    pub mod nn {
        pub use sptorch_nn::{generate_constrained, TokenTrie, GPT};
    }

    /// Stable optimizer API used by product runtimes.
    pub mod optim {
        pub use sptorch_optim::{clip_grad_norm, scale_gradients, zero_grad, AdamW, Optimizer, SGD};
    }

    /// Stable ops API used by product runtimes.
    pub mod ops {
        pub use sptorch_core_ops::cross_entropy_loss;
    }

    /// Stable checkpoint API used by product runtimes.
    pub mod checkpoint {
        pub use sptorch_serialize::{load_checkpoint, save_checkpoint};
    }

    /// Convenience prelude for product-side imports.
    pub mod prelude {
        pub use super::checkpoint::{load_checkpoint, save_checkpoint};
        pub use super::nn::{generate_constrained, TokenTrie, GPT};
        pub use super::ops::cross_entropy_loss;
        pub use super::optim::{clip_grad_norm, scale_gradients, zero_grad, AdamW, Optimizer, SGD};
    }
}
