# Python-kubernetes-scheduler-simulator
Python-kubernetes-scheduler-simulator

anu_kss_py/
├── configs/
│   ├── example_basic.yaml
│   ├── example_fgd.yaml
│   ├── example_spread.yaml
│   ├── example_gpupack.yaml
│   └── example_trace.yaml
├── workloads/                     (create this if using traces)
│   └── my_trace.csv
├── results/                       (created automatically)
├── sim/                           (your Python package)
│   ├── __init__.py
│   ├── pod.py
│   ├── node.py
│   ├── cluster.py
│   ├── policies.py
│   ├── scheduler.py
│   └── simulator.py
├── experiments.py
└── requirements.txt

-------------------------
# Gang Scheduling (FGD)

For FGD, instead of scheduling individual pods, you need to group pods by gang_id. In the arrival event, collect all pods of a gang that arrive within a short window (or all at once if the workload definition groups them). Then call fgd(gang_pods, cluster) to find a node, and if found, place all pods on that node (each with its own duration). If not, they stay pending and are retried periodically.

You’d need to modify _arrival to buffer gang pods until the gang is complete or a timeout occurs. The original simulator likely expects all gang pods to arrive at the same time (they are generated from a job specification). That makes it simpler.

-------------------------
# Installation (from the project root)

pip install -r requirements.txt


# Summary of Provided Files
File - Purpose
configs/small_test.yaml	 - Quick validation, minimal resources.
configs/medium_experiment.yaml - Realistic medium cluster with mixed workload.
configs/large_scale.yaml - Large cluster stress test (20 nodes, 2000 pods).
configs/trace_based.yaml - Uses a realistic CSV trace.
workloads/alibaba_style_trace.csv - 100‑pod trace with gang formations and varied resources.

```
python experiments.py --config configs/medium_experiment.yaml --output results/medium/ --plot
```



