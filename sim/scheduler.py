"""
scheduler.py

This module defines the Scheduler class, which is the core decision‑making
engine of the simulation. It takes pods from the pending queue, applies a
scheduling policy, and attempts to place them on suitable nodes.

Key responsibilities:
    - Schedule a single pod using the configured policy.
    - Schedule a gang of pods atomically using FGD (First Gang Deployed).
    - Schedule as many pending pods as possible (retry loop).
    - Group pending pods by gang ID for gang‑aware scheduling.

The Scheduler works closely with the Cluster, Node, and Pod data models,
and uses the policy functions defined in policies.py.

Usage example:
    cluster = Cluster(nodes)
    scheduler = Scheduler(cluster, policy="bestfit")
    # After adding pods to the pending queue:
    scheduled_count = scheduler.schedule_all_pending()
"""

from typing import List, Optional, Dict, Set, Tuple
import random
from collections import defaultdict

from .cluster import Cluster
from .pod import Pod
from .policies import get_policy, fgd, bestfit
from .node import Node


class Scheduler:
    """
    The main scheduling engine.

    Attributes:
        cluster (Cluster): The cluster state (nodes, pending pods, etc.).
        policy_name (str): The name of the active scheduling policy.
        _policy_func (callable): Cached reference to the policy function.
        rng (random.Random): Random number generator for stochastic policies.
    """

    def __init__(self, cluster: Cluster, policy_name: str = "bestfit",
                 seed: Optional[int] = None) -> None:
        """
        Initialise the scheduler with a cluster and a policy.

        Args:
            cluster: The Cluster object to schedule onto.
            policy_name: Name of the scheduling policy (e.g., "bestfit", "fgd").
            seed: Optional random seed for deterministic stochastic policies.
        """
        self.cluster = cluster
        self.policy_name = policy_name.lower().replace("-", "_")
        self._policy_func = None  # lazy-loaded
        self.rng = random.Random(seed) if seed is not None else random.Random()

        # Cache the policy function
        self._load_policy()

    def _load_policy(self) -> None:
        """Load the policy function from the registry."""
        try:
            self._policy_func = get_policy(self.policy_name)
        except ValueError as e:
            # Fallback to bestfit if the policy is unknown
            print(f"Warning: {e}. Falling back to 'bestfit'.")
            self._policy_func = bestfit
            self.policy_name = "bestfit"

    def set_policy(self, policy_name: str) -> None:
        """
        Change the scheduling policy at runtime.

        Args:
            policy_name: New policy name.
        """
        self.policy_name = policy_name.lower().replace("-", "_")
        self._load_policy()

    # -------------------------------------------------------------------------
    # Single‑Pod Scheduling
    # -------------------------------------------------------------------------

    def schedule_pod(self, pod: Pod) -> bool:
        """
        Attempt to schedule a single pod using the current policy.

        For gang policies (fgd), this method will treat the single pod as a
        gang of size 1 and call fgd([pod], cluster). For all other policies,
        it applies the policy function directly.

        Args:
            pod: The Pod to schedule. It must be in the "pending" state.

        Returns:
            True if the pod was successfully assigned to a node, False otherwise.
            If successful, the pod's state becomes "running" and node_name is set.
        """
        if pod.state != "pending":
            raise ValueError(f"Cannot schedule pod {pod.uid} – state is '{pod.state}', not 'pending'.")

        # If the pod is already assigned (should not happen), skip
        if pod.node_name is not None:
            return True

        # Special handling for FGD: even a single pod is treated as a gang
        if self.policy_name == "fgd":
            node = fgd([pod], self.cluster, self.rng)
        else:
            # Normal single‑pod policy
            node = self._policy_func(pod, self.cluster, self.rng)

        if node is None:
            return False

        # Assign the pod to the chosen node
        self.cluster.assign_pod_to_node(pod, node)
        return True

    # -------------------------------------------------------------------------
    # Gang Scheduling
    # -------------------------------------------------------------------------

    def schedule_gang(self, gang_pods: List[Pod]) -> bool:
        """
        Schedule a gang of pods atomically using the FGD policy.

        This method assumes that all pods in the list belong to the same gang
        and are all currently pending. It attempts to place all of them on a
        single node.

        Args:
            gang_pods: List of Pod objects belonging to the same gang.

        Returns:
            True if the entire gang was placed on one node, False otherwise.
            If successful, all pods are assigned to the same node; if not,
            none of the pods are assigned (atomicity).
        """
        if not gang_pods:
            return True  # empty gang is trivially scheduled

        # Ensure all pods are pending
        for p in gang_pods:
            if p.state != "pending":
                raise ValueError(f"Pod {p.uid} in gang is not pending (state: {p.state}).")

        # Check if they all share the same gang_id
        gang_id = gang_pods[0].gang_id
        for p in gang_pods[1:]:
            if p.gang_id != gang_id:
                raise ValueError(f"Mixed gang IDs in schedule_gang: {gang_id} vs {p.gang_id}")

        # Find a single node for the whole gang
        node = fgd(gang_pods, self.cluster, self.rng)
        if node is None:
            return False

        # Atomically assign all pods to the same node
        for p in gang_pods:
            self.cluster.assign_pod_to_node(p, node)
        return True

    # -------------------------------------------------------------------------
    # Batch Scheduling (Retry)
    # -------------------------------------------------------------------------

    def schedule_all_pending(self) -> int:
        """
        Attempt to schedule all currently pending pods.

        This is the main batch scheduling loop. It scans the pending queue
        and tries to place each pod. For gang scheduling (FGD), it groups
        pending pods by gang_id and schedules complete gangs atomically.

        Important: This method processes a snapshot of the pending queue.
        Pods that are successfully scheduled are removed from the queue;
        pods that still cannot fit remain in the queue for the next retry.

        Returns:
            The number of pods that were successfully scheduled in this call.
        """
        scheduled_count = 0

        # Get a snapshot of the pending queue (the cluster's pending list
        # will be modified as we assign pods, so we iterate over a copy)
        pending_copy = self.cluster.get_pending_pods().copy()

        if self.policy_name == "fgd":
            # Gang scheduling: group pending pods by gang_id
            gangs: Dict[str, List[Pod]] = defaultdict(list)
            non_gang_pods: List[Pod] = []

            for p in pending_copy:
                if p.gang_id:
                    gangs[p.gang_id].append(p)
                else:
                    non_gang_pods.append(p)

            # Schedule gangs first
            for gang_id, gang_pods in gangs.items():
                # Only schedule if all pods of the gang are still pending
                # (some might have been scheduled if they were in the copy but removed)
                still_pending = [p for p in gang_pods if p in self.cluster.pending_pods]
                if len(still_pending) == len(gang_pods):
                    if self.schedule_gang(still_pending):
                        scheduled_count += len(still_pending)
                else:
                    # Partial gang – reschedule the remaining
                    if still_pending:
                        # We can try to schedule the partial gang, but that's not FGD.
                        # We'll fall back to bestfit for the remaining pods individually.
                        for p in still_pending:
                            if self.schedule_pod(p):
                                scheduled_count += 1

            # Schedule non‑gang pods individually
            for p in non_gang_pods:
                if p in self.cluster.pending_pods:  # still pending
                    if self.schedule_pod(p):
                        scheduled_count += 1

        else:
            # Non‑gang policies: schedule pods one by one in FIFO order
            for p in pending_copy:
                if p in self.cluster.pending_pods:  # still pending
                    if self.schedule_pod(p):
                        scheduled_count += 1

        return scheduled_count

    # -------------------------------------------------------------------------
    # Retry Logic (for SimPy integration)
    # -------------------------------------------------------------------------

    def try_schedule_until_fixedpoint(self, max_attempts: int = 10) -> int:
        """
        Repeatedly attempt to schedule all pending pods until no more pods
        can be scheduled (fixed point) or until max_attempts is reached.

        This is useful after resources are released (e.g., pod completions)
        to retry previously unschedulable pods.

        Args:
            max_attempts: Maximum number of full sweeps over the pending queue.

        Returns:
            Total number of pods scheduled across all attempts.
        """
        total_scheduled = 0
        for _ in range(max_attempts):
            scheduled = self.schedule_all_pending()
            total_scheduled += scheduled
            if scheduled == 0:
                # No progress: break early
                break
        return total_scheduled

    # -------------------------------------------------------------------------
    # Statistics and Status
    # -------------------------------------------------------------------------

    def get_pending_gangs(self) -> Dict[str, int]:
        """
        Return a dictionary of gang_id -> number of pending pods for that gang.

        This is useful for monitoring gang‑aware scheduling progress.
        """
        gang_counts: Dict[str, int] = defaultdict(int)
        for p in self.cluster.pending_pods:
            if p.gang_id:
                gang_counts[p.gang_id] += 1
        return dict(gang_counts)

    def is_gang_ready(self, gang_id: str, expected_size: int) -> bool:
        """
        Check if a gang has all its members in the pending queue.
        This can be used to determine when a gang is fully submitted.

        Args:
            gang_id: The gang's unique identifier.
            expected_size: The number of pods that should be in this gang.

        Returns:
            True if the gang has exactly expected_size pending pods.
        """
        pending = [p for p in self.cluster.pending_pods if p.gang_id == gang_id]
        return len(pending) == expected_size

    # -------------------------------------------------------------------------
    # Debugging
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Scheduler(policy='{self.policy_name}', "
                f"pending={len(self.cluster.pending_pods)}, "
                f"running={self.cluster.get_running_pods_count()})")