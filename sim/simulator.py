"""
simulator.py

This module defines the Simulator class, which drives the discrete‑event
simulation using SimPy. It manages pod arrivals, scheduling, execution,
and completion events. The simulator integrates the Cluster, Scheduler,
and policies to model a Kubernetes‑like scheduling environment.

Key features:
    - Pod arrivals from a trace or generated workload.
    - Gang scheduling with atomic placement (FGD).
    - Automatic retry of pending pods after each completion.
    - Configurable runtime (service time) function.
    - Statistics collection: wait times, turnaround times, utilisation.
    - Logging and progress reporting.

Usage example:
    from sim.cluster import Cluster, Node
    from sim.pod import Pod, ResourceRequest
    from sim.scheduler import Scheduler
    from sim.simulator import Simulator

    # Build cluster, workload, scheduler...
    sim = Simulator(cluster, scheduler, workload_events)
    sim.run(until=10000)
    stats = sim.get_stats()
    print(stats)
"""

import simpy
import random
import logging
from typing import List, Optional, Dict, Any, Callable, Union, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from .cluster import Cluster
from .pod import Pod, ResourceRequest
from .scheduler import Scheduler
from .node import Node


# Default runtime function: exponential with mean 60 time units
def default_runtime_func(pod: Pod, rng: random.Random) -> float:
    """
    Default runtime (service time) model: exponential distribution with mean 60.

    Args:
        pod: The pod being executed.
        rng: Random number generator.

    Returns:
        Service time in simulation time units.
    """
    return rng.expovariate(1.0 / 60.0)


# Alternative runtime: based on resource request (CPU + memory/2 + GPU*5)
def resource_based_runtime(pod: Pod, rng: random.Random) -> float:
    """
    Runtime proportional to resource requests: base 10 + cpu*5 + memory/4 + gpu*20.

    Args:
        pod: The pod.
        rng: Random generator (not used here, but kept for signature).

    Returns:
        Service time in simulation time units.
    """
    cpu = pod.resources.cpu
    mem = pod.resources.memory
    gpu = pod.resources.gpu
    return 10.0 + cpu * 5.0 + mem * 2.0 + gpu * 20.0


@dataclass
class SimulationStats:
    """Container for simulation statistics."""
    total_pods_submitted: int = 0
    total_pods_scheduled: int = 0
    total_pods_completed: int = 0
    total_pods_failed: int = 0
    wait_times: List[float] = field(default_factory=list)
    run_times: List[float] = field(default_factory=list)
    turnaround_times: List[float] = field(default_factory=list)
    cluster_utilization_history: List[Tuple[float, Dict[str, float]]] = field(
        default_factory=list
    )
    # Per-node utilization can be added if needed


class Simulator:
    """
    Discrete‑event simulator for scheduling policies.

    Attributes:
        env (simpy.Environment): The SimPy simulation environment.
        cluster (Cluster): The cluster state.
        scheduler (Scheduler): The scheduling engine.
        workload_events (List[Tuple[float, Union[Pod, List[Pod]]]]):
            List of (submit_time, pod_or_gang) events.
        runtime_func (Callable): Function to compute pod runtime.
        rng (random.Random): Random generator.
        stats (SimulationStats): Accumulated statistics.
        log_interval (float): If set, log progress every N time units.
    """

    def __init__(
        self,
        cluster: Cluster,
        scheduler: Scheduler,
        workload_events: List[Tuple[float, Union[Pod, List[Pod]]]],
        runtime_func: Optional[Callable[[Pod, random.Random], float]] = None,
        seed: Optional[int] = None,
        log_interval: Optional[float] = None,
    ) -> None:
        """
        Initialise the simulator.

        Args:
            cluster: The Cluster object.
            scheduler: The Scheduler object (already configured with a policy).
            workload_events: List of (submit_time, pod_or_list) events.
                             If the value is a single Pod, it's a normal pod.
                             If it's a list of Pods, they are treated as a gang.
            runtime_func: Function that takes a Pod and a random.Random and
                          returns the service time (float). If None, uses
                          default exponential(mean=60).
            seed: Random seed for reproducibility.
            log_interval: If set, print progress messages every log_interval
                          simulation time units.
        """
        self.env = simpy.Environment()
        self.cluster = cluster
        self.scheduler = scheduler
        self.workload_events = sorted(workload_events, key=lambda x: x[0])
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.runtime_func = runtime_func or default_runtime_func
        self.log_interval = log_interval
        self.stats = SimulationStats()
        self._last_log_time = 0.0

        # Internal state for pending gang tracking
        self._pending_gang_pods: Dict[str, List[Pod]] = defaultdict(list)

    # -------------------------------------------------------------------------
    # Core Event Handlers
    # -------------------------------------------------------------------------

    def _arrival(self, submit_time: float, pod_or_gang: Union[Pod, List[Pod]]) -> None:
        """
        Handle a pod or gang arrival event.

        This is a SimPy process that waits until the submit_time and then
        adds the pod(s) to the cluster's pending queue, attempts scheduling,
        and triggers a retry if they cannot be placed immediately.

        Args:
            submit_time: The time when the pod(s) should be submitted.
            pod_or_gang: Single Pod or list of Pods (gang).
        """
        yield self.env.timeout(submit_time - self.env.now)

        self.stats.total_pods_submitted += 1 if isinstance(pod_or_gang, Pod) else len(pod_or_gang)

        if isinstance(pod_or_gang, Pod):
            pod = pod_or_gang
            pod.submit_time = self.env.now
            self.cluster.add_pending_pod(pod)
            # Try to schedule immediately
            scheduled = self.scheduler.schedule_pod(pod)
            if scheduled:
                self.stats.total_pods_scheduled += 1
                # Schedule completion
                self.env.process(self._run_pod(pod))
            else:
                # Pod remains pending; will be retried later
                pass
        else:
            # Gang: all pods are submitted at the same time
            gang_pods = pod_or_gang
            gang_id = gang_pods[0].gang_id
            if not gang_id:
                raise ValueError("Gang pods must have a gang_id set.")
            # Set submit_time for all
            for p in gang_pods:
                p.submit_time = self.env.now
                self.cluster.add_pending_pod(p)
            # Attempt to schedule the whole gang atomically
            success = self.scheduler.schedule_gang(gang_pods)
            if success:
                self.stats.total_pods_scheduled += len(gang_pods)
                # Schedule each pod's completion
                for p in gang_pods:
                    self.env.process(self._run_pod(p))
            else:
                # Gang not schedulable; pods remain pending
                pass

        # After any scheduling attempt, try to schedule as many as possible
        # (this is also done on completion, but we do it here for immediate
        # retry of pods that might have been blocked by this arrival)
        self._retry_pending()

    def _run_pod(self, pod: Pod) -> None:
        """
        Simulate the execution of a pod and its completion.

        This process waits for the runtime (service time), then removes the pod
        from the cluster, updates statistics, and triggers a retry of pending pods.

        Args:
            pod: The Pod that is running.
        """
        # Get the service time
        runtime = self.runtime_func(pod, self.rng)

        # Wait for completion
        yield self.env.timeout(runtime)

        # Pod finishes
        pod.finish_time = self.env.now

        # Remove from node
        try:
            self.cluster.remove_pod_from_node(pod)
        except ValueError as e:
            logging.warning(f"Error removing pod {pod.uid}: {e}")
            # If removal fails (should not happen), mark as failed
            pod.state = "failed"
            self.stats.total_pods_failed += 1
            return

        # Update statistics
        self.stats.total_pods_completed += 1
        wait_time = pod.start_time - pod.submit_time
        run_time = pod.finish_time - pod.start_time
        turnaround = pod.finish_time - pod.submit_time
        self.stats.wait_times.append(wait_time)
        self.stats.run_times.append(run_time)
        self.stats.turnaround_times.append(turnaround)

        # Retry pending pods now that resources are free
        self._retry_pending()

    def _retry_pending(self) -> None:
        """
        Attempt to schedule all currently pending pods (and gangs).

        This is called after pod completions and after new arrivals.
        It uses the scheduler's fixed‑point retry logic.
        """
        # We can call scheduler.try_schedule_until_fixedpoint() to attempt
        # scheduling all pending pods. It will return the number newly scheduled.
        newly_scheduled = self.scheduler.try_schedule_until_fixedpoint(max_attempts=10)

        if newly_scheduled > 0:
            # For each newly scheduled pod, start its completion process
            # (they are now in the running state)
            # We need to find which pods got scheduled. Since the scheduler
            # may have scheduled multiple, we can scan the cluster's nodes
            # and find pods that have start_time set but no finish_time yet.
            # However, this is a bit tricky; we could store a callback list.
            # Simpler: the scheduler's schedule_pod/schedule_gang already
            # started the _run_pod process for each scheduled pod. However,
            # _retry_pending is called from _arrival or _run_pod, which already
            # starts processes for immediate scheduling. For retries, we need
            # to start them here.

            # To avoid duplicate processes, we must check if a pod already
            # has a running process. We'll use a flag on the pod or maintain
            # a set of pods that have been started. For simplicity, we'll
            # start the process for all running pods that don't have a finish_time.
            for node in self.cluster.nodes:
                for p in node.pods:
                    if p.state == "running" and p.finish_time is None:
                        # This pod is running but hasn't been scheduled for completion yet.
                        # Check if it already has a process (we can't easily know).
                        # We'll use a pod attribute `_process_started` to avoid duplicates.
                        if not hasattr(p, '_process_started') or not p._process_started:
                            p._process_started = True
                            self.env.process(self._run_pod(p))

            # The above is a bit hacky but works. Alternatively, we could have the
            # scheduler return the list of scheduled pods and start processes there.

    # -------------------------------------------------------------------------
    # Main Run Method
    # -------------------------------------------------------------------------

    def run(self, until: float) -> None:
        """
        Run the simulation until the specified time.

        This schedules all arrival events and then starts the SimPy event loop.
        It also optionally logs progress at regular intervals.

        Args:
            until: The simulation time to run until.
        """
        # Schedule all arrival events
        for submit_time, pod_or_gang in self.workload_events:
            if submit_time < until:  # only schedule if within horizon
                self.env.process(self._arrival(submit_time, pod_or_gang))

        # Optional progress logging
        if self.log_interval is not None:
            self.env.process(self._log_progress())

        # Run the simulation
        self.env.run(until=until)

        # Final log
        if self.log_interval is not None:
            self._print_progress(self.env.now, final=True)

    def _log_progress(self) -> None:
        """
        Periodic logging of simulation progress.
        """
        while True:
            yield self.env.timeout(self.log_interval)
            if self.env.now > self._last_log_time:
                self._print_progress(self.env.now)

    def _print_progress(self, current_time: float, final: bool = False) -> None:
        """
        Print a progress message with current statistics.
        """
        pending = self.cluster.get_pending_pods_count()
        running = self.cluster.get_running_pods_count()
        completed = self.stats.total_pods_completed
        total_submitted = self.stats.total_pods_submitted
        util = self.cluster.get_cluster_utilization()
        avg_wait = sum(self.stats.wait_times) / len(self.stats.wait_times) if self.stats.wait_times else 0.0
        avg_turnaround = sum(self.stats.turnaround_times) / len(self.stats.turnaround_times) if self.stats.turnaround_times else 0.0

        status = "FINAL" if final else "PROGRESS"
        print(f"[{status} at t={current_time:.2f}] "
              f"Submitted={total_submitted}, Scheduled={completed+pending+running}, "
              f"Completed={completed}, Pending={pending}, Running={running}, "
              f"AvgWait={avg_wait:.2f}, AvgTurnaround={avg_turnaround:.2f}, "
              f"Util: CPU={util['cpu']:.2f}, Mem={util['memory']:.2f}, GPU={util['gpu']:.2f}")

        self._last_log_time = current_time

    # -------------------------------------------------------------------------
    # Statistics Collection
    # -------------------------------------------------------------------------

    def get_stats(self) -> SimulationStats:
        """
        Return the accumulated simulation statistics.

        Returns:
            SimulationStats object with all recorded metrics.
        """
        return self.stats

    def get_utilization_history(self) -> List[Tuple[float, Dict[str, float]]]:
        """
        Return the history of cluster utilisation over time.

        To populate this, we need to sample at intervals. For now, we only
        provide the final utilisation, but we can extend to record history
        by adding a periodic sampling event.

        This method returns the current list (which may be empty if not sampled).
        """
        return self.stats.cluster_utilization_history

    # -------------------------------------------------------------------------
    # Helper to load workload from a trace (e.g., YAML)
    # -------------------------------------------------------------------------

    @staticmethod
    def load_workload_from_trace(trace: List[Dict[str, Any]]) -> List[Tuple[float, Union[Pod, List[Pod]]]]:
        """
        Convert a trace (list of dicts) into the workload_events format.

        Expected trace format:
            [
                {
                    "time": 0.0,
                    "pod": {
                        "uid": "pod-1",
                        "name": "nginx",
                        "resources": {"cpu": 1.0, "memory": 2.0, "gpu": 0},
                        "gang_id": "gang-A"  # optional
                    }
                },
                ...
            ]

        If a gang_id is present and multiple pods have the same gang_id at the
        same time, they will be grouped. However, this function groups by
        (time, gang_id) automatically.

        Args:
            trace: List of dictionaries, each with "time" and "pod" keys.

        Returns:
            List of (submit_time, pod_or_gang) events.
        """
        # Group by (time, gang_id) to form gangs
        from collections import defaultdict
        grouped: Dict[Tuple[float, str], List[Pod]] = defaultdict(list)
        single_pods: List[Tuple[float, Pod]] = []

        for entry in trace:
            t = entry["time"]
            pod_dict = entry["pod"]
            gang_id = pod_dict.get("gang_id")
            pod = Pod(
                uid=pod_dict.get("uid", f"pod-{len(single_pods)}"),
                name=pod_dict.get("name", "pod"),
                resources=ResourceRequest(
                    cpu=pod_dict["resources"]["cpu"],
                    memory=pod_dict["resources"]["memory"],
                    gpu=pod_dict["resources"]["gpu"],
                ),
                namespace=pod_dict.get("namespace", "default"),
                gang_id=gang_id,
                priority=pod_dict.get("priority", 0),
            )
            if gang_id:
                grouped[(t, gang_id)].append(pod)
            else:
                single_pods.append((t, pod))

        events = []
        for (t, gang_id), gang_pods in grouped.items():
            events.append((t, gang_pods))
        events.extend(single_pods)
        return sorted(events, key=lambda x: x[0])