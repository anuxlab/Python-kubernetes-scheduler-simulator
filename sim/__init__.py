"""
sim - A Python-based simulation framework for Kubernetes scheduler policies.

This package is a Python port of the anu_kss (ANU Kubernetes Scheduler Simulator)
framework. It provides discrete-event simulation of pod scheduling, resource
management, and gang scheduling policies (FGD, best-fit, spread, GPU-packing).

Main components:
    - Cluster: Represents the entire simulated Kubernetes cluster.
    - Node: Represents a single worker node with CPU, memory, and GPU capacity.
    - Pod: Represents a workload unit with resource requests and lifecycle state.
    - ResourceRequest: A simple container for CPU, memory, and GPU quantities.
    - Scheduler: The scheduling engine that applies policies to pending pods.
    - Simulator: The SimPy-based discrete-event simulation engine.

Policies implemented:
    - bestfit:  Pack pods onto nodes with the least remaining capacity.
    - spread:   Spread pods across nodes to balance load.
    - gpupack:  Pack GPU workloads on nodes already using GPUs.
    - fgd:      First Gang Deployed – place all pods of a gang on one node.
    - random:   Random selection among feasible nodes (baseline).

Usage example:
    from sim import Cluster, Node, Pod, ResourceRequest, Scheduler, Simulator

    # Build cluster
    nodes = [Node("node-1", ResourceRequest(4.0, 8.0, 1)), ...]
    cluster = Cluster(nodes)

    # Create workload
    pod = Pod(uid="pod-1", resources=ResourceRequest(1.0, 2.0, 0))

    # Set up scheduler and simulator
    scheduler = Scheduler(cluster, policy="bestfit")
    sim = Simulator(cluster, scheduler, [(0.0, pod)])
    sim.run(until=1000)

    # Get statistics
    stats = sim.get_stats()
    print(f"Average wait time: {sum(stats.wait_times)/len(stats.wait_times):.2f}")
"""

# -------------------------------------------------------------------------
# Version
# -------------------------------------------------------------------------

__version__ = "0.1.0"

# -------------------------------------------------------------------------
# Core Data Models
# -------------------------------------------------------------------------

from .pod import Pod, ResourceRequest
from .node import Node
from .cluster import Cluster

# -------------------------------------------------------------------------
# Scheduling Engine
# -------------------------------------------------------------------------

from .scheduler import Scheduler
from .policies import (
    bestfit,
    spread,
    gpupack,
    fgd,
    random_policy,
    get_policy,
    POLICY_REGISTRY,
    schedule_pod,
)

# -------------------------------------------------------------------------
# Simulation Engine
# -------------------------------------------------------------------------

from .simulator import Simulator, SimulationStats, default_runtime_func, resource_based_runtime

# -------------------------------------------------------------------------
# Public API (what is exported with `from sim import *`)
# -------------------------------------------------------------------------

__all__ = [
    # Core data models
    "Pod",
    "ResourceRequest",
    "Node",
    "Cluster",

    # Scheduler and policies
    "Scheduler",
    "bestfit",
    "spread",
    "gpupack",
    "fgd",
    "random_policy",
    "get_policy",
    "POLICY_REGISTRY",
    "schedule_pod",

    # Simulator
    "Simulator",
    "SimulationStats",
    "default_runtime_func",
    "resource_based_runtime",

    # Package metadata
    "__version__",
]

# -------------------------------------------------------------------------
# Optional: Configure logging for the package
# -------------------------------------------------------------------------

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())