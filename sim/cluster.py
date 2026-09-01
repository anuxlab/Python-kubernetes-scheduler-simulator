"""
cluster.py

This module defines the Cluster class, which represents the entire simulated
Kubernetes cluster. It maintains a list of Node objects, a queue of pending
Pods, and provides methods to query resource availability, assign pods to
nodes, and release resources when pods complete.

The cluster is the central state object used by the scheduling policies
(e.g., best-fit, spread, FGD) and the SimPy simulation loop.
"""

from collections import defaultdict

from .node import Node

# Import the core data types from the other modules
from .pod import Pod, ResourceRequest


class Cluster:
    """
    A simulated Kubernetes cluster consisting of worker nodes.

    Attributes:
        nodes (List[Node]): All nodes in the cluster.
        pending_pods (List[Pod]): FIFO queue of pods waiting to be scheduled.
        _pod_to_node (Dict[str, str]): Quick lookup: pod UID -> node name.
        _gang_to_pods (Dict[str, List[Pod]]): Quick lookup: gang ID -> list of pods.
    """

    def __init__(self, nodes: list[Node]) -> None:
        """
        Initialise the cluster with a list of pre-configured nodes.

        Args:
            nodes: A list of Node objects representing the cluster's capacity.
        """
        self.nodes = nodes
        self.pending_pods: list[Pod] = []

        # Internal indexes for fast lookups
        self._pod_to_node: dict[str, str] = {}
        self._gang_to_pods: dict[str, list[Pod]] = defaultdict(list)

        # Validate that node names are unique
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("Node names must be unique.")

    # -------------------------------------------------------------------------
    # Pending Pod Management
    # -------------------------------------------------------------------------

    def add_pending_pod(self, pod: Pod) -> None:
        """
        Add a pod to the pending queue. Also indexes it by its gang ID
        (if any) to support gang‑aware scheduling.

        Args:
            pod: The Pod object to enqueue.
        """
        if pod.uid in self._pod_to_node:
            raise ValueError(f"Pod {pod.uid} is already in the cluster.")
        self.pending_pods.append(pod)
        if pod.gang_id:
            self._gang_to_pods[pod.gang_id].append(pod)

    def get_pending_pods(self) -> list[Pod]:
        """Return a shallow copy of the current pending pod queue."""
        return self.pending_pods.copy()

    def get_pending_pods_by_gang(self, gang_id: str) -> list[Pod]:
        """
        Return all pending pods that belong to a specific gang.

        This is used by the FGD (First Gang Deployed) policy to collect
        all pods of a gang and try to place them together on one node.

        Args:
            gang_id: The unique identifier of the gang.

        Returns:
            A list of Pod objects belonging to that gang, in the order
            they were added (typically the order defined in the workload).
        """
        return self._gang_to_pods.get(gang_id, []).copy()

    def remove_pending_pod(self, pod: Pod) -> None:
        """
        Remove a pod from the pending queue (e.g., after it has been scheduled
        or if it fails permanently). Also cleans up the gang index.

        Args:
            pod: The Pod to remove.
        """
        if pod in self.pending_pods:
            self.pending_pods.remove(pod)
        if pod.gang_id and pod in self._gang_to_pods.get(pod.gang_id, []):
            self._gang_to_pods[pod.gang_id].remove(pod)
            if not self._gang_to_pods[pod.gang_id]:
                del self._gang_to_pods[pod.gang_id]

    # -------------------------------------------------------------------------
    # Node Lookup and Resource Queries
    # -------------------------------------------------------------------------

    def get_node_by_name(self, name: str) -> Node | None:
        """
        Retrieve a node by its unique name.

        Args:
            name: The node's name.

        Returns:
            The Node object, or None if no such node exists.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_all_nodes(self) -> list[Node]:
        """Return the full list of nodes."""
        return self.nodes.copy()

    def get_available_nodes_for_pod(self, pod: Pod) -> list[Node]:
        """
        Return a list of all nodes that currently have enough free resources
        to accommodate the given pod.

        This is a generic helper used by scheduling policies to filter the
        set of feasible nodes.

        Args:
            pod: The Pod requesting resources.

        Returns:
            A list of Node objects where the pod can fit.
        """
        return [n for n in self.nodes if n.can_fit(pod.resources)]

    def can_fit_gang_on_node(self, gang_pods: list[Pod], node: Node) -> bool:
        """
        Check whether a single node can host all pods of a gang simultaneously.

        This sums the resource requests of all pods in the gang and compares
        against the node's available capacity. It is used specifically by the
        FGD policy.

        Args:
            gang_pods: List of Pod objects belonging to the same gang.
            node: The Node to test.

        Returns:
            True if the node has sufficient free CPU, memory, and GPU to run
            all gang pods together; False otherwise.
        """
        total_req = ResourceRequest(cpu=0.0, memory=0.0, gpu=0)
        for p in gang_pods:
            total_req.cpu += p.resources.cpu
            total_req.memory += p.resources.memory
            total_req.gpu += p.resources.gpu

        # Check against the node's *currently available* resources
        available_cpu = node.capacity.cpu - node.allocated.cpu
        available_mem = node.capacity.memory - node.allocated.memory
        available_gpu = node.capacity.gpu - node.allocated.gpu

        return (
            total_req.cpu <= available_cpu
            and total_req.memory <= available_mem
            and total_req.gpu <= available_gpu
        )

    # -------------------------------------------------------------------------
    # Pod Assignment and Resource Accounting
    # -------------------------------------------------------------------------

    def assign_pod_to_node(self, pod: Pod, node: Node) -> None:
        """
        Assign a pod to a specific node. This updates:
        - The node's allocated resources.
        - The node's list of running pods.
        - The pod's state and node_name field.
        - The internal _pod_to_node index.

        This method assumes that the node has already been checked for
        sufficient capacity (e.g., via can_fit()).

        Args:
            pod: The Pod to assign.
            node: The target Node.

        Raises:
            ValueError: If the pod is already assigned to another node, or if
                        the node cannot fit the pod.
        """
        if pod.uid in self._pod_to_node:
            existing_node = self.get_node_by_name(self._pod_to_node[pod.uid])
            raise ValueError(
                f"Pod {pod.uid} is already assigned to node {existing_node.name if existing_node else 'unknown'}."
            )

        if not node.can_fit(pod.resources):
            raise ValueError(
                f"Node {node.name} does not have enough resources for pod {pod.uid}."
            )

        # Update the node
        node.add_pod(pod)

        # Update the pod
        pod.node_name = node.name
        pod.state = "running"

        # Update internal index
        self._pod_to_node[pod.uid] = node.name

        # If this pod was pending, remove it from the pending queue
        if pod in self.pending_pods:
            self.remove_pending_pod(pod)

    def remove_pod_from_node(self, pod: Pod) -> None:
        """
        Release the resources of a finished/failed pod from its assigned node.
        This is called when a pod completes its execution.

        Args:
            pod: The Pod to remove (must have a node_name set).

        Raises:
            ValueError: If the pod is not found on any node.
        """
        if pod.uid not in self._pod_to_node:
            raise ValueError(f"Pod {pod.uid} is not assigned to any node.")

        node_name = self._pod_to_node[pod.uid]
        node = self.get_node_by_name(node_name)
        if node is None:
            raise ValueError(f"Node {node_name} no longer exists in cluster.")

        # Remove from node
        node.remove_pod(pod)

        # Update pod state
        pod.node_name = None
        pod.state = "succeeded"  # or "failed", caller can override

        # Clean internal index
        del self._pod_to_node[pod.uid]

    def get_node_for_pod(self, pod: Pod) -> Node | None:
        """
        Return the Node to which the pod is currently assigned, or None
        if it is not assigned.

        Args:
            pod: The Pod to look up.

        Returns:
            The Node object, or None.
        """
        node_name = self._pod_to_node.get(pod.uid)
        if node_name:
            return self.get_node_by_name(node_name)
        return None

    # -------------------------------------------------------------------------
    # Aggregate Statistics and Utilities
    # -------------------------------------------------------------------------

    def get_total_resources(self) -> ResourceRequest:
        """
        Compute the total resource capacity of the entire cluster.

        Returns:
            A ResourceRequest object with the sum of all nodes' capacities.
        """
        total = ResourceRequest(cpu=0.0, memory=0.0, gpu=0)
        for node in self.nodes:
            total.cpu += node.capacity.cpu
            total.memory += node.capacity.memory
            total.gpu += node.capacity.gpu
        return total

    def get_used_resources(self) -> ResourceRequest:
        """
        Compute the total resources currently allocated across all nodes.

        Returns:
            A ResourceRequest object with the sum of all allocated resources.
        """
        used = ResourceRequest(cpu=0.0, memory=0.0, gpu=0)
        for node in self.nodes:
            used.cpu += node.allocated.cpu
            used.memory += node.allocated.memory
            used.gpu += node.allocated.gpu
        return used

    def get_cluster_utilization(self) -> dict[str, float]:
        """
        Return a dictionary with overall cluster utilisation fractions
        for CPU, memory, and GPU.

        Returns:
            A dict with keys 'cpu', 'memory', 'gpu' and values between 0 and 1.
        """
        total = self.get_total_resources()
        used = self.get_used_resources()
        return {
            "cpu": used.cpu / total.cpu if total.cpu > 0 else 0.0,
            "memory": used.memory / total.memory if total.memory > 0 else 0.0,
            "gpu": used.gpu / total.gpu if total.gpu > 0 else 0.0,
        }

    def get_running_pods_count(self) -> int:
        """Return the total number of pods currently running across all nodes."""
        return sum(len(node.pods) for node in self.nodes)

    def get_pending_pods_count(self) -> int:
        """Return the number of pods waiting in the pending queue."""
        return len(self.pending_pods)

    # -------------------------------------------------------------------------
    # Debugging and Validation
    # -------------------------------------------------------------------------

    def validate_invariants(self) -> bool:
        """
        Perform a sanity check to ensure the cluster state is consistent.
        This is useful for debugging and testing.

        Checks:
        - Each node's allocated resources equal the sum of its pods' requests.
        - Every assigned pod appears in the _pod_to_node index and vice versa.
        - No pod appears both in pending and on a node.

        Returns:
            True if all invariants hold, otherwise raises an AssertionError.
        """
        # 1. Check node resource accounting
        for node in self.nodes:
            sum_cpu = sum(p.resources.cpu for p in node.pods)
            sum_mem = sum(p.resources.memory for p in node.pods)
            sum_gpu = sum(p.resources.gpu for p in node.pods)
            assert abs(node.allocated.cpu - sum_cpu) < 1e-6, (
                f"Node {node.name}: allocated CPU mismatch"
            )
            assert abs(node.allocated.memory - sum_mem) < 1e-6, (
                f"Node {node.name}: allocated memory mismatch"
            )
            assert abs(node.allocated.gpu - sum_gpu) < 1e-6, (
                f"Node {node.name}: allocated GPU mismatch"
            )

        # 2. Check pod-to-node index consistency
        assigned_uids = set()
        for node in self.nodes:
            for pod in node.pods:
                assert pod.uid in self._pod_to_node, (
                    f"Pod {pod.uid} on node {node.name} missing from index"
                )
                assert self._pod_to_node[pod.uid] == node.name, (
                    f"Pod {pod.uid} index points to {self._pod_to_node[pod.uid]} but is on {node.name}"
                )
                assigned_uids.add(pod.uid)

        # 3. All indexed pods must be on a node
        for uid, node_name in self._pod_to_node.items():
            node = self.get_node_by_name(node_name)
            assert node is not None, f"Index points to non-existent node {node_name}"
            assert any(p.uid == uid for p in node.pods), (
                f"Pod {uid} in index but not in node's pod list"
            )

        # 4. No pod in both pending and running
        pending_uids = {p.uid for p in self.pending_pods}
        overlap = pending_uids & assigned_uids
        assert not overlap, f"Pods {overlap} are both pending and running"

        return True
