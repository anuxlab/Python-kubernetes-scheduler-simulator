# Python-kubernetes-scheduler-simulator
Python-kubernetes-scheduler-simulator

Python-kubernetes-scheduler-simulator/
├── .github/
│   └── workflows/
│       ├── ci.yml           # Run tests on every commit
│       └── publish.yml      # Build and push Docker image
├── docker/
│   ├── Dockerfile           # Production Docker image
│   └── docker-compose.yml   # Easy orchestration
├── tests/
│   ├── test_policies.py
│   ├── test_scheduler.py
│   └── test_simulator.py
├── docs/
│   ├── cluster-config.md
│   ├── scheduler-config.md
│   ├── workload-config.md
│   └── trace-format.md
├── data/
│   ├── raw/                 # Production traces (large files, .gitignore)
│   └── prepare_trace.py     
├── experiments/
│   └── batch_run.py         
├── analysis/
│   └── plot_results.py      
├── sim/                     
├── configs/                 
├── workloads/               
├── main.py                  
├── experiments.py           
├── requirements.txt         
└── README.md                

-------------------------
# Gang Scheduling (FGD)

For FGD, instead of scheduling individual pods, you need to group pods by gang_id. In the arrival event, collect all pods of a gang that arrive within a short window (or all at once if the workload definition groups them). Then call fgd(gang_pods, cluster) to find a node, and if found, place all pods on that node (each with its own duration). If not, they stay pending and are retried periodically.

You’d need to modify _arrival to buffer gang pods until the gang is complete or a timeout occurs. The original simulator likely expects all gang pods to arrive at the same time (they are generated from a job specification). That makes it simpler.

-------------------------
# Installation (from the project root)

pip install -r requirements.txt

# Run a single experiment from a unified YAML

python main.py run --config configs/basic.yaml --output results/basic/ --plot

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

# What You Already Have
- Core simulation engine (simulator.py, scheduler.py, policies.py).
- Data models (pod.py, node.py, cluster.py).
- A single‑experiment runner (experiments.py) with YAML configs.
- Synthetic workload generator and a CSV trace loader.

# What This CLI Provides
- Consistent interface – mimics the simon apply command structure.
- Flexibility – both unified and separate config modes.
- Reproducibility – seed can be set from command line or config.
- Output management – creates a structured results directory with CSV, JSON, and optional plots.
- Integration – uses your existing sim package without modification.


--------------------------------------
# Docker

Building and Running
1. Build the Docker image

From the project root directory:
bash

docker build -f docker/Dockerfile -t anuxlab/python-k8s-simulator:latest .

Or using docker-compose (from the docker/ folder):
bash

cd docker
docker-compose build

2. Run a single experiment
bash

docker run --rm -v $(pwd)/configs:/app/configs -v $(pwd)/workloads:/app/workloads -v $(pwd)/results:/app/results anuxlab/python-k8s-simulator:latest run --config configs/basic.yaml --output results/docker_run --plot

3. Run with docker-compose (recommended)
bash

cd docker
docker-compose up

This will:

    Build the image (if not already built).

    Mount the configs/, workloads/, and results/ directories from your host.

    Run the simulator with the command specified in docker-compose.yml (by default run --config configs/basic.yaml ...).

To override the command temporarily:
bash

docker-compose run --rm simulator run --config configs/medium_experiment.yaml --output results/medium_docker --seed 123

4. Run an interactive shell inside the container (for debugging)
bash

docker run --rm -it anuxlab/python-k8s-simulator:latest /bin/bash

📦 Publishing to Docker Hub (Optional)

To share your image:
bash

# Tag with your Docker Hub username
docker tag anuxlab/python-k8s-simulator:latest yourusername/python-k8s-simulator:latest

# Push
docker push yourusername/python-k8s-simulator:latest

Then users can run:
bash

docker run --rm -v $(pwd)/configs:/app/configs -v $(pwd)/results:/app/results yourusername/python-k8s-simulator:latest run --config configs/my_exp.yaml --output results/

🧪 Testing with a Sample Command

After building, test with a simple command:
bash

docker run --rm anuxlab/python-k8s-simulator:latest run --help

Expected output: the help text of the run subcommand.
