use std::sync::OnceLock;

use tokio::sync::broadcast;
use versioning::{EvolutionMetrics, FenceState, HardwareState, VersionNode};

#[derive(Debug, Clone)]
pub enum LiveEvolutionEvent {
    Metrics(EvolutionMetrics),
    VersionCommit(VersionNode),
    Fence(FenceState),
    HardwareState(HardwareState),
}

static EVENT_BUS: OnceLock<broadcast::Sender<LiveEvolutionEvent>> = OnceLock::new();

fn bus() -> &'static broadcast::Sender<LiveEvolutionEvent> {
    EVENT_BUS.get_or_init(|| {
        let (tx, _rx) = broadcast::channel(1024);
        tx
    })
}

pub fn publish(event: LiveEvolutionEvent) {
    let _ = bus().send(event);
}

pub fn subscribe() -> broadcast::Receiver<LiveEvolutionEvent> {
    bus().subscribe()
}
