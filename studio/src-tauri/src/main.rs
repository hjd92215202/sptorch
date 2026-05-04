#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, RwLock};

use sptorch_studio::engine_bridge::{
    bootstrap_default_storage, get_engine_status, start_evolution_stream, trigger_atomic_swap, AppState,
    EngineBridge,
};
use versioning::EvolutionMetrics;

fn main() {
    let storage = Arc::new(RwLock::new(bootstrap_default_storage()));
    let bridge = Arc::new(EngineBridge::new(storage));

    // Simulated runtime metric stream for dashboard testing.
    let bridge_clone = bridge.clone();
    tauri::async_runtime::spawn(async move {
        let mut step = 0u32;
        loop {
            step = step.wrapping_add(1);
            let accum_target = 8;
            let accum_current = step % accum_target;
            let grad_scale = if step % 17 == 0 { 0.55 } else { 1.0 + ((step % 5) as f32) * 0.03 };

            bridge_clone.push_metric(EvolutionMetrics {
                ts_ms: (step as u64) * 100,
                loss: 2.0 / ((step + 1) as f32),
                grad_norm: (step as f32).sin().abs() + 0.1,
                grad_scale_factor: grad_scale,
                accum_current,
                accum_target,
                version_id: bridge_clone.snapshot_status().map(|s| s.active_version).unwrap_or(1),
                fence: None,
            });

            tokio::time::sleep(tokio::time::Duration::from_millis(350)).await;
        }
    });

    tauri::Builder::default()
        .manage(AppState { bridge })
        .invoke_handler(tauri::generate_handler![get_engine_status, start_evolution_stream, trigger_atomic_swap])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
