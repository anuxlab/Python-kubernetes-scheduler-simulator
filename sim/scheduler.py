"""
sim/scheduler.py – Scheduler with plugin‑based scoring (updated).
"""

import random
from collections import defaultdict

from .cluster import Cluster
from .config import KubeSchedulerConfiguration
from .node import Node
from .plugins import instantiate_plugin
from .pod import Pod, ResourceRequest


class Scheduler:
    """
    Scheduler that uses a list of scoring plugins to select the best node.
    """

    def __init__(
        self,
        cluster: Cluster,
        policy: str | None = None,
        config: KubeSchedulerConfiguration | None = None,
        seed: int | None = None,
    ):
        """
        Initialise the scheduler.

        Args:
            cluster: The cluster state.
            policy: Legacy policy name (e.g., "bestfit"). If given, it creates
                    a single plugin of the corresponding type with weight 1.
            config: KubeSchedulerConfiguration object with plugins and weights.
            seed: Random seed for stochastic plugins.
        """
        self.cluster = cluster
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self._plugins: list[tuple] = []  # (plugin_instance, weight)

        if config:
            # Use plugin configuration
            plugins_with_weights = config.get_plugins_with_weights()
            for p in plugins_with_weights:
                plugin_args = config.get_plugin_args(p.name)
                plugin_instance = instantiate_plugin(
                    p.name, args=plugin_args, seed=seed
                )
                self._plugins.append((plugin_instance, p.weight))
        elif policy:
            # Legacy mode: convert policy name to a single plugin
            plugin_name = self._policy_to_plugin(policy)
            plugin_instance = instantiate_plugin(plugin_name, seed=seed)
            self._plugins.append((plugin_instance, 1))
        else:
            # Default: use BestFitScore
            plugin_instance = instantiate_plugin("BestFitScore", seed=seed)
            self._plugins.append((plugin_instance, 1))

        if not self._plugins:
            raise ValueError("No plugins configured for the scheduler.")

    @staticmethod
    def _policy_to_plugin(policy: str) -> str:
        """Map legacy policy names to plugin names."""
        mapping = {
            "bestfit": "BestFitScore",
            "spread": "SpreadScore",
            "gpupack": "GpuPackingScore",
            "fgd": "FGDScore",
            "random": "RandomScore",
        }
        return mapping.get(policy.lower(), "BestFitScore")

    def _compute_node_score(self, pod: Pod, node: Node) -> float:
        """Compute the total weighted score for a pod on a node."""
        total = 0.0
        for plugin, weight in self._plugins:
            score = plugin.Score(pod, node, self.cluster)
            total += score * weight
        return total

    def _compute_gang_node_score(self, gang_pods: list[Pod], node: Node) -> float:
        """
        Compute the total weighted score for a gang on a node.
        For FGDScore, we need to check if the node can fit all pods.
        """
        total = 0.0
        for plugin, weight in self._plugins:
            if hasattr(plugin, "ScoreGang"):
                score = plugin.ScoreGang(gang_pods, node, self.cluster)
            else:
                # For non‑gang plugins, we need to aggregate scores over pods?
                # Simplification: treat the gang as a single pod with total resources.
                total_req = ResourceRequest(0.0, 0.0, 0)
                for p in gang_pods:
                    total_req.cpu += p.resources.cpu
                    total_req.memory += p.resources.memory
                    total_req.gpu += p.resources.gpu
                temp_pod = Pod(uid="tmp", name="tmp", resources=total_req)
                score = plugin.Score(temp_pod, node, self.cluster)
            total += score * weight
        return total

    def schedule_pod(self, pod: Pod) -> bool:
        """
        Schedule a single pod using the plugin‑based scoring.
        Returns True if scheduled successfully.
        """
        if pod.state != "pending":
            raise ValueError(f"Cannot schedule pod {pod.uid} – state is '{pod.state}'")

        # Get feasible nodes
        feasible = [n for n in self.cluster.nodes if n.can_fit(pod.resources)]
        if not feasible:
            return False

        # Score each node
        best_node = None
        best_score = -float("inf")
        for node in feasible:
            score = self._compute_node_score(pod, node)
            # Tie‑break by node name for determinism
            if score > best_score or (
                score == best_score
                and (best_node is None or node.name < best_node.name)
            ):
                best_score = score
                best_node = node

        if best_node is None:
            return False

        self.cluster.assign_pod_to_node(pod, best_node)
        return True

    def schedule_gang(self, gang_pods: list[Pod]) -> bool:
        """
        Schedule a gang of pods atomically using plugin‑based scoring.
        Returns True if the entire gang is placed on one node.
        """
        if not gang_pods:
            return True

        # Sum resources of all gang pods
        total_req = ResourceRequest(0.0, 0.0, 0)
        for p in gang_pods:
            total_req.cpu += p.resources.cpu
            total_req.memory += p.resources.memory
            total_req.gpu += p.resources.gpu

        # Find feasible nodes (that can fit the whole gang)
        feasible = []
        for node in self.cluster.nodes:
            free = node.get_free_resources()
            if (
                total_req.cpu <= free.cpu
                and total_req.memory <= free.memory
                and total_req.gpu <= free.gpu
            ):
                feasible.append(node)

        if not feasible:
            return False

        # Score each feasible node
        best_node = None
        best_score = -float("inf")
        for node in feasible:
            score = self._compute_gang_node_score(gang_pods, node)
            if score > best_score or (
                score == best_score
                and (best_node is None or node.name < best_node.name)
            ):
                best_score = score
                best_node = node

        if best_node is None:
            return False

        # Assign all pods to the best node
        for p in gang_pods:
            self.cluster.assign_pod_to_node(p, best_node)
        return True

    # -------------------------------------------------------------------------
    # Batch scheduling (unchanged)
    # -------------------------------------------------------------------------

    def schedule_all_pending(self) -> int:
        """Schedule all pending pods/gangs using the plugin‑based scoring."""
        scheduled_count = 0
        pending_copy = self.cluster.get_pending_pods().copy()

        # Group pending pods by gang_id
        gangs = defaultdict(list)
        non_gang_pods = []
        for p in pending_copy:
            if p.gang_id:
                gangs[p.gang_id].append(p)
            else:
                non_gang_pods.append(p)

        # Schedule gangs first
        for gang_pods in gangs.values():
            still_pending = [p for p in gang_pods if p in self.cluster.pending_pods]
            if len(still_pending) == len(gang_pods) and self.schedule_gang(
                still_pending
            ):
                scheduled_count += len(still_pending)

        # Schedule non‑gang pods
        for p in non_gang_pods:
            if p in self.cluster.pending_pods and self.schedule_pod(p):
                scheduled_count += 1

        return scheduled_count

    def try_schedule_until_fixedpoint(self, max_attempts: int = 10) -> int:
        """Retry scheduling all pending pods until no progress."""
        total_scheduled = 0
        for _ in range(max_attempts):
            scheduled = self.schedule_all_pending()
            total_scheduled += scheduled
            if scheduled == 0:
                break
        return total_scheduled
