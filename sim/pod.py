"""
pod.py

This module defines the core data types for pods and resource requests
in the Kubernetes simulation.

Classes:
    ResourceRequest: A simple container for CPU, memory, and GPU quantities.
    Pod: Represents a single pod with its resource requirements, lifecycle
         state, and scheduling metadata.

These classes are used by the Cluster, Node, and Scheduler modules.
They are designed to be lightweight and easy to serialise for logging.
"""

import uuid
from dataclasses import dataclass


@dataclass
class ResourceRequest:
    """
    A resource request specifying CPU, memory, and GPU quantities.

    Attributes:
        cpu (float): CPU cores requested (can be fractional, e.g., 0.5).
        memory (float): Memory in gigabytes (GiB).
        gpu (int): Number of GPUs requested.
    """

    cpu: float
    memory: float
    gpu: int

    def __post_init__(self) -> None:
        """Validate that resource values are non‑negative."""
        if self.cpu < 0:
            raise ValueError(f"CPU request cannot be negative: {self.cpu}")
        if self.memory < 0:
            raise ValueError(f"Memory request cannot be negative: {self.memory}")
        if self.gpu < 0:
            raise ValueError(f"GPU request cannot be negative: {self.gpu}")

    def __add__(self, other: "ResourceRequest") -> "ResourceRequest":
        """Add two resource requests together (useful for summing gang pods)."""
        return ResourceRequest(
            cpu=self.cpu + other.cpu,
            memory=self.memory + other.memory,
            gpu=self.gpu + other.gpu,
        )

    def __sub__(self, other: "ResourceRequest") -> "ResourceRequest":
        """Subtract one resource request from another."""
        return ResourceRequest(
            cpu=self.cpu - other.cpu,
            memory=self.memory - other.memory,
            gpu=self.gpu - other.gpu,
        )

    def is_zero(self) -> bool:
        """Check if all resources are zero."""
        return self.cpu == 0 and self.memory == 0 and self.gpu == 0


@dataclass
class Pod:
    """
    Represents a single pod in the simulation.

    Attributes:
        uid (str): Unique identifier for the pod.
        name (str): Human‑readable name (usually from the workload file).
        namespace (str): Kubernetes namespace (default "default").
        resources (ResourceRequest): Resource requirements.
        gang_id (Optional[str]): Gang identifier if this pod belongs to a gang,
                                  otherwise None.
        node_name (Optional[str]): The name of the node this pod is assigned to,
                                   or None if pending.
        submit_time (float): Simulation time when the pod was submitted.
        start_time (Optional[float]): Simulation time when the pod started running.
        finish_time (Optional[float]): Simulation time when the pod finished.
        state (str): One of "pending", "running", "succeeded", "failed", or "completed".
        priority (int): Optional priority (default 0) for future QoS extensions.
    """

    uid: str
    name: str
    resources: ResourceRequest
    namespace: str = "default"
    gang_id: str | None = None
    node_name: str | None = None
    submit_time: float = 0.0
    start_time: float | None = None
    finish_time: float | None
    state: str = "pending"
    priority: int = 0  # not used yet, but reserved

    # Use __slots__ to reduce memory overhead for many pods
    __slots__ = (
        "finish_time",
        "gang_id",
        "name",
        "namespace",
        "node_name",
        "priority",
        "resources",
        "start_time",
        "state",
        "submit_time",
        "uid",
    )

    def __post_init__(self) -> None:
        """Validate pod fields and set default UID if not provided."""
        if not self.uid:
            # Generate a pseudo-UUID if none given (useful for test workloads)
            self.uid = f"pod-{uuid.uuid4().hex[:8]}"
        if not self.name:
            self.name = self.uid
        if self.state not in {"pending", "running", "succeeded", "failed", "completed"}:
            raise ValueError(f"Invalid pod state: {self.state}")

    def is_pending(self) -> bool:
        """Return True if the pod is waiting for scheduling."""
        return self.state == "pending"

    def is_running(self) -> bool:
        """Return True if the pod is currently running."""
        return self.state == "running"

    def is_terminated(self) -> bool:
        """Return True if the pod has finished (succeeded, failed, or completed)."""
        return self.state in {"succeeded", "failed", "completed"}

    def get_wait_time(self, current_time: float) -> float:
        """
        Compute how long the pod has been waiting since submission.

        Args:
            current_time: The current simulation time.

        Returns:
            Waiting duration in seconds (simulation time units).
            If start_time is set, returns start_time - submit_time;
            otherwise returns current_time - submit_time.
        """
        if self.start_time is not None:
            return self.start_time - self.submit_time
        else:
            return current_time - self.submit_time

    def get_run_time(self) -> float | None:
        """
        Return the duration the pod ran, if it has finished.

        Returns:
            The run duration (finish_time - start_time), or None if not finished.
        """
        if self.start_time is not None and self.finish_time is not None:
            return self.finish_time - self.start_time
        return None

    def __repr__(self) -> str:
        """Compact representation for logging."""
        return (
            f"Pod(uid='{self.uid}', name='{self.name}', "
            f"resources={self.resources}, state='{self.state}')"
        )
