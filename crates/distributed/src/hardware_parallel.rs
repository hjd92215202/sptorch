//! Hardware-aware parallel validation plans.
//!
//! The distributed crate owns orchestration over a hardware topology supplied by
//! `sptorch-hal`. These plans are dry-run friendly: they can validate multi-board
//! Tank9k connectivity before a real serial/PCIe backend exists.

use sptorch_hal::topology::{AllReduceEstimate, HardwareTopology, MatmulPartitionPlan, TopologyValidation};
use sptorch_hal::DeviceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelCollective {
    RingAllReduce,
    Matmul2DShard,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HardwareParallelPlan {
    pub topology_name: String,
    pub world: Vec<DeviceId>,
    pub validation: TopologyValidation,
    pub collectives: Vec<ParallelCollective>,
    pub allreduce: Option<AllReduceEstimate>,
    pub matmul: Option<MatmulPartitionPlan>,
}

impl HardwareParallelPlan {
    pub fn ready(&self) -> bool {
        self.validation.connected && self.validation.total_nodes > 0
    }

    pub fn summary(&self) -> String {
        format!(
            "topology={} world={} connected={} collectives={}",
            self.topology_name,
            self.world.len(),
            self.validation.connected,
            self.collectives.len()
        )
    }
}

pub fn plan_ring_allreduce(topology: &HardwareTopology, payload_bytes: usize) -> HardwareParallelPlan {
    let validation = topology.validate_connectivity();
    let world = topology.ring_plan();
    let allreduce = topology.allreduce_cost_estimate(payload_bytes);
    HardwareParallelPlan {
        topology_name: topology.name.clone(),
        world,
        validation,
        collectives: vec![ParallelCollective::RingAllReduce],
        allreduce,
        matmul: None,
    }
}

pub fn plan_tank9k_matmul_validation(
    topology: &HardwareTopology,
    m: usize,
    k: usize,
    n: usize,
    gradient_payload_bytes: usize,
) -> HardwareParallelPlan {
    let validation = topology.validate_connectivity();
    let world = topology.ring_plan();
    let allreduce = topology.allreduce_cost_estimate(gradient_payload_bytes);
    let matmul = Some(topology.matmul_partition_plan(m, k, n));
    HardwareParallelPlan {
        topology_name: topology.name.clone(),
        world,
        validation,
        collectives: vec![ParallelCollective::Matmul2DShard, ParallelCollective::RingAllReduce],
        allreduce,
        matmul,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sptorch_hal::topology::{HardwareLink, HardwareNode, LinkRole, TransportKind};

    fn tank9k_ring(n: usize) -> HardwareTopology {
        let mut topo = HardwareTopology::new("tank9k-validation-ring");
        for i in 0..n {
            topo.add_node(HardwareNode::new(
                DeviceId::tank9k(i),
                format!("tank9k-{i}"),
                "tank9k",
                4096,
            ));
        }
        for i in 0..n {
            topo.add_link(HardwareLink::new(
                DeviceId::tank9k(i),
                DeviceId::tank9k((i + 1) % n),
                TransportKind::Serial,
                LinkRole::Data,
                32,
                150,
            ));
        }
        topo
    }

    #[test]
    fn builds_multi_board_tank9k_validation_plan() {
        let topo = tank9k_ring(4);
        let plan = plan_tank9k_matmul_validation(&topo, 32, 32, 32, 32 * 32 * 4);

        assert!(plan.ready(), "{}", plan.summary());
        assert_eq!(plan.world.len(), 4);
        assert_eq!(
            plan.collectives,
            vec![ParallelCollective::Matmul2DShard, ParallelCollective::RingAllReduce]
        );
        assert_eq!(plan.matmul.as_ref().unwrap().shards.len(), 4);
        assert_eq!(plan.allreduce.as_ref().unwrap().ring_hops, 4);
    }

    #[test]
    fn broken_topology_produces_not_ready_plan() {
        let mut topo = HardwareTopology::new("broken-tank9k");
        topo.add_node(HardwareNode::new(DeviceId::tank9k(0), "a", "tank9k", 4096));
        topo.add_node(HardwareNode::new(DeviceId::tank9k(1), "b", "tank9k", 4096));

        let plan = plan_ring_allreduce(&topo, 4096);
        assert!(!plan.ready());
        assert!(plan.allreduce.is_none());
        assert!(!plan.validation.diagnostics.is_empty());
    }
}
