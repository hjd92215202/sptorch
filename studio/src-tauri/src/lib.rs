pub mod engine_bridge;

use std::sync::{Arc, RwLock};

use engine_bridge::{
    bootstrap_default_storage, get_engine_status, start_evolution_stream, trigger_atomic_swap, AppState, EngineBridge,
};

pub fn run() {
    let storage = Arc::new(RwLock::new(bootstrap_default_storage()));
    let bridge = Arc::new(EngineBridge::new(storage));

    tauri::Builder::default()
        .manage(AppState { bridge })
        .invoke_handler(tauri::generate_handler![
            get_engine_status,
            start_evolution_stream,
            trigger_atomic_swap
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
