"""
policies.py

This module implements the core scheduling policies used in the anu_kss
Kubernetes scheduler simulator. Each policy is a function that takes a
Cluster and a Pod (or gang of Pods) and returns a suitable Node, or None
if no node can accommodate the request.

Policies implemented:
    - bestfit:  Pack pods onto the node with the least remaining capacity
                (minimises fragmentation).
    - spread:   Spread pods across nodes to balance load (choose node with
                fewest running pods).
    - gpupack:  Prioritise packing GPU workloads on nodes that already have
                GPUs allocated.
    - fgd:      First Gang Deployed – place all pods of a gang on one node.
    - random:   Random selection among feasible nodes (baseline).

Usage:
    node = bestfit(pod, cluster)
    if node:
        cluster.assign_pod_to_node(pod, node)

For gang scheduling:
    node = fgd(gang_pods, cluster)
    if node:
        for p in gang_pods:
            cluster.assign_pod_to_node(p, node)
"""

import random
from collections.abc import Callable

from .cluster import Cluster
from .node import Node
from .pod import Pod, ResourceRequest

# -------------------------------------------------------------------------
# Non-Gang Policies
# -------------------------------------------------------------------------


def bestfit(
    pod: Pod, cluster: Cluster, rng: random.Random | None = None
) -> Node | None:
    """
    Best-fit scheduling: place the pod on the feasible node that has the
    least remaining resources after placing the pod (i.e., tightest fit).

    This minimises fragmentation and is commonly used for CPU/memory
    intensive workloads. It scores each node by the sum of remaining
    resource fractions (CPU + memory + GPU) after placing the pod, and
    selects the node with the smallest score.

    If there is a tie, the node with the smaller name is chosen for
    deterministic behaviour.

    Args:
        pod: The pod to schedule.
        cluster: The cluster state.
        rng: Optional random generator (not used but kept for consistency).

    Returns:
        The chosen Node, or None if no node can fit the pod.
    """
    best_node = None
    best_score = float("inf")

    for node in cluster.nodes:
        if not node.can_fit(pod.resources):
            continue

        # Compute remaining resources after adding the pod
        remaining_cpu = node.capacity.cpu - node.allocated.cpu - pod.resources.cpu
        remaining_mem = (
            node.capacity.memory - node.allocated.memory - pod.resources.memory
        )
        remaining_gpu = node.capacity.gpu - node.allocated.gpu - pod.resources.gpu

        # Normalise by total capacity to get fractional scores
        cpu_frac = remaining_cpu / node.capacity.cpu if node.capacity.cpu > 0 else 0.0
        mem_frac = (
            remaining_mem / node.capacity.memory if node.capacity.memory > 0 else 0.0
        )
        gpu_frac = remaining_gpu / node.capacity.gpu if node.capacity.gpu > 0 else 0.0

        # Score = sum of fractions (lower is tighter)
        score = cpu_frac + mem_frac + gpu_frac

        # Tie‑break by node name for determinism
        if score < best_score or (
            score == best_score and (best_node is None or node.name < best_node.name)
        ):
            best_score = score
            best_node = node

    return best_node


def spread(pod: Pod, cluster: Cluster, rng: random.Random | None = None) -> Node | None:
    """
    Spread scheduling: place the pod on the feasible node that currently
    has the fewest running pods. This balances load across the cluster.

    This policy is common for web services that want to avoid hotspots.
    If multiple nodes have the same pod count, the one with the most free
    resources is chosen (as a secondary tie‑breaker).

    Args:
        pod: The pod to schedule.
        cluster: The cluster state.
        rng: Optional random generator (not used).

    Returns:
        The chosen Node, or None if no node can fit the pod.
    """
    candidates = []
    for node in cluster.nodes:
        if node.can_fit(pod.resources):
            candidates.append(node)

    if not candidates:
        return None

    # Find the minimum number of running pods among candidates
    min_pods = min(len(node.pods) for node in candidates)

    # Filter to those with the minimum pod count
    best_candidates = [n for n in candidates if len(n.pods) == min_pods]

    # Among those, choose the one with the most free resources (sum of CPU+memory+GPU)
    def free_resource_sum(node: Node) -> float:
        free = node.get_free_resources()
        return free.cpu + free.memory + free.gpu

    # Sort by free resource sum descending, then by name for determinism
    best_candidates.sort(key=lambda n: (-free_resource_sum(n), n.name))
    return best_candidates[0]


def gpupack(
    pod: Pod, cluster: Cluster, rng: random.Random | None = None
) -> Node | None:
    """
    GPU‑packing scheduling: for pods that request GPUs, try to place them on
    nodes that already have GPU usage, to concentrate GPU workloads and
    leave non‑GPU nodes for CPU‑only pods. If the pod does not request GPUs,
    it falls back to best‑fit.

    Among feasible nodes with at least one allocated GPU, choose the one with
    the highest GPU utilisation (i.e., pack GPUs tightly). If no node has
    GPUs allocated, choose the node with the most free GPU capacity.

    Args:
        pod: The pod to schedule.
        cluster: The cluster state.
        rng: Optional random generator (not used).

    Returns:
        The chosen Node, or None if no node can fit the pod.
    """
    if pod.resources.gpu == 0:
        # If no GPU requested, use best‑fit
        return bestfit(pod, cluster)

    # First, get all feasible nodes
    feasible = [n for n in cluster.nodes if n.can_fit(pod.resources)]
    if not feasible:
        return None

    # Divide into nodes that already have GPUs allocated vs. those that don't
    gpu_nodes = [n for n in feasible if n.allocated.gpu > 0]
    non_gpu_nodes = [n for n in feasible if n.allocated.gpu == 0]

    # Prefer nodes with GPUs already allocated (to pack)
    if gpu_nodes:
        # Sort by GPU utilisation descending (pack tightest), then by name
        gpu_nodes.sort(
            key=lambda n: (
                n.allocated.gpu / n.capacity.gpu if n.capacity.gpu > 0 else 0.0
            ),
            reverse=True,
        )
        # Tie‑break by node name
        gpu_nodes.sort(key=lambda n: n.name)
        return gpu_nodes[0]  # after reverse sort, highest utilisation first
    else:
        # No GPU nodes, pick the one with most free GPU capacity
        non_gpu_nodes.sort(
            key=lambda n: (n.get_free_resources().gpu, n.name), reverse=True
        )
        return non_gpu_nodes[0]


def random_policy(
    pod: Pod, cluster: Cluster, rng: random.Random | None = None
) -> Node | None:
    """
    Random scheduling: randomly pick a feasible node.

    This is a baseline policy for comparison and for randomisation in
    experiments.

    Args:
        pod: The pod to schedule.
        cluster: The cluster state.
        rng: A random.Random instance (if None, uses a default seed).

    Returns:
        A randomly chosen Node from the feasible set, or None if none.
    """
    if rng is None:
        rng = random  # use global random

    feasible = [n for n in cluster.nodes if n.can_fit(pod.resources)]
    if not feasible:
        return None
    return rng.choice(feasible)


# -------------------------------------------------------------------------
# Gang Scheduling Policy
# -------------------------------------------------------------------------


def fgd(
    gang_pods: list[Pod], cluster: Cluster, rng: random.Random | None = None
) -> Node | None:
    """
    First Gang Deployed (FGD) scheduling: place all pods of a gang together
    on a single node. This is a hard constraint: the node must have enough
    free resources to accommodate the sum of all pods' requests.

    FGD is the core policy from the anu_kss paper for gang scheduling.
    It scans nodes and returns the first node that can fit the entire gang,
    or None if no such node exists. The order of nodes is as given in
    the cluster's nodes list.

    Args:
        gang_pods: List of Pod objects belonging to the same gang.
                   They must have the same gang_id.
        cluster: The cluster state.
        rng: Optional random generator (not used).

    Returns:
        The first Node that can fit the entire gang, or None.
    """
    if not gang_pods:
        return None

    # Sum resources of all gang pods
    total_req = ResourceRequest(cpu=0.0, memory=0.0, gpu=0)
    for p in gang_pods:
        total_req.cpu += p.resources.cpu
        total_req.memory += p.resources.memory
        total_req.gpu += p.resources.gpu

    # Scan nodes in order
    for node in cluster.nodes:
        # Check if the node can fit the combined request
        free = node.get_free_resources()
        if (
            total_req.cpu <= free.cpu
            and total_req.memory <= free.memory
            and total_req.gpu <= free.gpu
        ):
            return node

    return None


def dotproduct(
    pod: Pod, cluster: Cluster, rng: random.Random | None = None
) -> Node | None:
    """
    Dot‑product scoring: score = pod_request · node_free_resources.
    The node with the highest dot product is selected.
    This is a standard baseline policy used in the HKUST simulator.

    Args:
        pod: The pod to schedule.
        cluster: The cluster state.
        rng: Optional random generator (not used).

    Returns:
        The node with the highest dot‑product score, or None if no node can fit.
    """
    best_node = None
    best_score = -float("inf")

    for node in cluster.nodes:
        if not node.can_fit(pod.resources):
            continue

        free = node.get_free_resources()
        # Dot product of pod request and free resources
        score = (
            pod.resources.cpu * free.cpu
            + pod.resources.memory * free.memory
            + pod.resources.gpu * free.gpu
        )

        # Tie‑break by node name for determinism
        if score > best_score or (
            score == best_score and (best_node is None or node.name < best_node.name)
        ):
            best_score = score
            best_node = node

    return best_node


def gpuclustering(
    pod: Pod, cluster: Cluster, rng: random.Random | None = None
) -> Node | None:
    """
    GPU Clustering: concentrate GPU workloads on nodes that already have GPU allocations.
    For pods requesting GPUs: pick the feasible node with the *most* allocated GPUs.
    For pods not requesting GPUs: pick the feasible node with the *least* allocated GPUs
    (preferring nodes with zero GPU usage).

    This policy aims to separate GPU and non‑GPU workloads, improving utilisation
    and reducing fragmentation.

    Args:
        pod: The pod to schedule.
        cluster: The cluster state.
        rng: Optional random generator (not used).

    Returns:
        The selected node, or None if no feasible node.
    """
    feasible = [n for n in cluster.nodes if n.can_fit(pod.resources)]
    if not feasible:
        return None

    if pod.resources.gpu > 0:
        # For GPU pods: prefer nodes with highest GPU allocation (clustering)
        # Sort by allocated GPUs descending, then by node name for determinism
        feasible.sort(key=lambda n: (n.allocated.gpu, n.name), reverse=True)
    else:
        # For non‑GPU pods: prefer nodes with lowest GPU allocation (avoid GPU nodes)
        feasible.sort(key=lambda n: (n.allocated.gpu, n.name))
        # Additionally, if there are nodes with zero GPU, we pick the first of those
        # But sorting already handles that.

    return feasible[0]  # after sorting, the first node matches the criterion


# -------------------------------------------------------------------------
# Policy Registry
# -------------------------------------------------------------------------

# A dictionary mapping policy names (as used in configuration) to functions.
# This allows easy look-up and dynamic selection.
POLICY_REGISTRY: dict[str, Callable] = {
    "bestfit": bestfit,
    "spread": spread,
    "gpupack": gpupack,
    "fgd": fgd,
    "random": random_policy,
    # Also include aliases for convenience
    "best-fit": bestfit,
    "best_fit": bestfit,
    "firstfit": None,  # not implemented explicitly; fallback to bestfit
    "dotproduct": dotproduct,
    "gpuclustering": gpuclustering,
}


def get_policy(name: str) -> Callable:
    """
    Retrieve a scheduling policy function by its name.

    Args:
        name: Policy name (case-insensitive, underscores/hyphens allowed).

    Returns:
        The policy function.

    Raises:
        ValueError: If the policy name is not recognised.
    """
    # Normalise: lower-case, replace hyphens with underscores
    norm = name.lower().replace("-", "_")
    policy = POLICY_REGISTRY.get(norm)
    if policy is None:
        # Fallback: try first-fit (which we treat as bestfit)
        if norm in ("firstfit", "first-fit", "first_fit"):
            return bestfit
        raise ValueError(
            f"Unknown policy name: '{name}'. Available: {list(POLICY_REGISTRY.keys())}"
        )
    return policy


# -------------------------------------------------------------------------
# Helper: schedule a pod using any policy
# -------------------------------------------------------------------------


def schedule_pod(
    pod: Pod, cluster: Cluster, policy_name: str, rng: random.Random | None = None
) -> Node | None:
    """
    Convenience function to schedule a single pod using a named policy.

    For gang policies (fgd), this will not work correctly because it
    requires a list of pods. Use fgd() directly for gang scheduling.

    Args:
        pod: The pod to schedule.
        cluster: The cluster state.
        policy_name: Name of the policy (e.g., "bestfit").
        rng: Optional random generator for stochastic policies.

    Returns:
        The selected Node, or None.
    """
    policy = get_policy(policy_name)
    if policy_name.lower() == "fgd":
        # fgd expects a list, but we only have one pod; treat as single-pod gang
        return fgd([pod], cluster, rng)
    return policy(pod, cluster, rng)
