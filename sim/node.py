"""
node.py

This module defines the Node class, which represents a single worker node
in the simulated Kubernetes cluster. Each node has a fixed capacity of
CPU, memory, and GPU resources, and tracks how much of those resources
are currently allocated to running pods.

The Node class provides methods to:
- Check if a pod can fit (can_fit)
- Add a pod (add_pod) – updates allocated resources and pod list
- Remove a pod (remove_pod) – releases resources
- Query remaining resources and utilisation

The class uses slots to reduce memory overhead, which is important when
simulating large clusters with thousands of nodes.
"""

from dataclasses import dataclass, field

# Import the ResourceRequest and Pod classes from the pod module
# (These will be defined in pod.py)
from .pod import Pod, ResourceRequest


@dataclass
class Node:
    """
    A simulated Kubernetes node with resource capacity and allocation.

    Attributes:
        name (str): Unique identifier for the node.
        capacity (ResourceRequest): Total resources the node can provide.
        allocated (ResourceRequest): Resources currently used by running pods.
        pods (List[Pod]): List of Pod objects currently scheduled on this node.
    """

    # Using __slots__ reduces memory overhead and speeds up attribute access.
    # This is especially beneficial when simulating many nodes.
    __slots__ = ("allocated", "capacity", "name", "pods")

    name: str
    capacity: ResourceRequest
    allocated: ResourceRequest = field(
        default_factory=lambda: ResourceRequest(0.0, 0.0, 0)
    )
    pods: list[Pod] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Perform validation after initialisation.

        Raises:
            ValueError: If node name is empty, or if capacity has negative values.
        """
        if not self.name:
            raise ValueError("Node name cannot be empty.")
        if self.capacity.cpu < 0 or self.capacity.memory < 0 or self.capacity.gpu < 0:
            raise ValueError("Node capacity cannot be negative.")
        # Ensure allocated starts at zero (if not explicitly set)
        if (
            self.allocated.cpu != 0
            or self.allocated.memory != 0
            or self.allocated.gpu != 0
        ):
            # If someone passed a non-zero allocated, we allow it, but warn or reset?
            # We'll reset to zero to avoid inconsistent state.
            self.allocated = ResourceRequest(0.0, 0.0, 0)

    # -------------------------------------------------------------------------
    # Resource Capacity and Availability
    # -------------------------------------------------------------------------

    def can_fit(self, request: ResourceRequest) -> bool:
        """
        Check if the node has enough free resources to accommodate the given request.

        This is the primary feasibility check used by scheduling policies.

        Args:
            request: ResourceRequest containing CPU, memory, and GPU needs.

        Returns:
            True if all resources are available, False otherwise.
        """
        return (
            self.allocated.cpu + request.cpu <= self.capacity.cpu
            and self.allocated.memory + request.memory <= self.capacity.memory
            and self.allocated.gpu + request.gpu <= self.capacity.gpu
        )

    def get_free_resources(self) -> ResourceRequest:
        """
        Compute the currently available (free) resources on the node.

        Returns:
            A ResourceRequest object representing unused CPU, memory, and GPU.
        """
        return ResourceRequest(
            cpu=self.capacity.cpu - self.allocated.cpu,
            memory=self.capacity.memory - self.allocated.memory,
            gpu=self.capacity.gpu - self.allocated.gpu,
        )

    def get_utilization(self) -> dict:
        """
        Return the utilisation fraction for each resource type.

        Returns:
            A dict with keys 'cpu', 'memory', 'gpu' and values between 0 and 1.
        """
        return {
            "cpu": self.allocated.cpu / self.capacity.cpu
            if self.capacity.cpu > 0
            else 0.0,
            "memory": self.allocated.memory / self.capacity.memory
            if self.capacity.memory > 0
            else 0.0,
            "gpu": self.allocated.gpu / self.capacity.gpu
            if self.capacity.gpu > 0
            else 0.0,
        }

    # -------------------------------------------------------------------------
    # Pod Management
    # -------------------------------------------------------------------------

    def add_pod(self, pod: Pod) -> None:
        """
        Schedule a pod on this node. This updates:
        - The node's allocated resources.
        - The node's list of running pods.
        - The pod's node_name and state (set to "running").

        This method assumes that can_fit() has already been called and returned True.
        It does not check feasibility again for performance reasons.

        Args:
            pod: The Pod to add.

        Raises:
            ValueError: If the pod is already on this node (duplicate) or if
                        the pod would exceed capacity (should not happen if
                        can_fit was checked).
        """
        # Prevent double-scheduling
        if pod in self.pods:
            raise ValueError(f"Pod {pod.uid} is already running on node {self.name}.")

        # Optional safety check (can be removed for performance)
        if not self.can_fit(pod.resources):
            raise ValueError(
                f"Node {self.name} cannot fit pod {pod.uid} (CPU: {pod.resources.cpu}, "
                f"Memory: {pod.resources.memory}, GPU: {pod.resources.gpu}). "
                f"Free: CPU={self.get_free_resources().cpu}, "
                f"Memory={self.get_free_resources().memory}, "
                f"GPU={self.get_free_resources().gpu}."
            )

        # Update node resources
        self.allocated.cpu += pod.resources.cpu
        self.allocated.memory += pod.resources.memory
        self.allocated.gpu += pod.resources.gpu

        # Add pod to the list
        self.pods.append(pod)

        # Update pod state
        pod.node_name = self.name
        pod.state = "running"

    def remove_pod(self, pod: Pod) -> None:
        """
        Remove a pod from the node, releasing its allocated resources.
        This is called when a pod finishes (succeeds or fails).

        Args:
            pod: The Pod to remove.

        Raises:
            ValueError: If the pod is not found on this node.
        """
        if pod not in self.pods:
            raise ValueError(f"Pod {pod.uid} is not running on node {self.name}.")

        # Release resources
        self.allocated.cpu -= pod.resources.cpu
        self.allocated.memory -= pod.resources.memory
        self.allocated.gpu -= pod.resources.gpu

        # Remove from list
        self.pods.remove(pod)

        # Update pod state (caller may set to "succeeded" or "failed")
        pod.node_name = None
        pod.state = "completed"  # generic state, can be overridden

    def get_pod_by_uid(self, uid: str) -> Pod | None:
        """
        Retrieve a pod running on this node by its unique identifier.

        Args:
            uid: The pod's UID.

        Returns:
            The Pod object if found, otherwise None.
        """
        for pod in self.pods:
            if pod.uid == uid:
                return pod
        return None

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        """Human-readable representation for debugging."""
        return (
            f"Node(name='{self.name}', capacity={self.capacity}, "
            f"allocated={self.allocated}, pods={len(self.pods)})"
        )

    def __str__(self) -> str:
        """Shorter representation."""
        return f"Node({self.name})"
