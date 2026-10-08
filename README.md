# Bivariate Reliability Degradation Index (RDI)

## Abstract
Intelligent systems deployed in resource-constrained edge computing environments operate under continuous environmental variability. When input data degrades due to environmental noise or distribution shifts, systems suffer a dual penalty: feature extraction algorithms struggle, leading to a sharp decline in predictive accuracy, and this struggle triggers a massive spike in computational latency causing thermal throttling and memory bottlenecks.

This repository implements a **Bivariate Reliability Degradation Index (RDI)** grounded in algorithmic state-machine logic. Instead of standard evaluation metrics that treat model inference as an isolated mathematical event under static conditions, this framework utilizes a 5-state Non-deterministic Finite Automaton (NFA) to track system degradation across both **accuracy** and **latency**. By measuring the Time-to-Failure (TTF) in discrete operational frames under hardware constraints, developers can practically evaluate exact operational limits before deploying machine learning models into decentralized physical environments.

## Repository Prototypes

The project contains two operational prototypes validating this approach:

### 1. `RDI_Prototype_Bivariate.py` (NFA State-Transition Evaluation)
The updated prototype implementing the formal **Bivariate NFA State Model** outlined in the research.
- **Dataset**: Dynamically streams the internet-hosted **UCI HIGGS dataset (11 million rows)** without exhausting local edge node memory. 
- **Methodology**: Evaluates each inference block as a discrete "frame," applying algorithmic synthetic noise. It independently tracks latency spikes and accuracy drops, resolving the system's operational health into one of five strictly defined algorithmic states:
  - `State 0`: Optimal Operation
  - `State 1`: Minor Degradation
  - `State 2`: Critical Degradation
  - `State 3 (qf)`: Terminal Failure (Single Metric Failure)
  - `State 4 (q_fatal)`: Fatal Lockup (Simultaneous Failure)
- **Time-To-Failure**: Quantifies system survival by logging the consecutive frames successfully processed before hitting an absorbing failure state.

### 2. `RDI_Prototype1.py` (Standard Degradation Script)
The original 5-component prototype focusing on image classification workloads.
- **Dataset**: Designed for local computer vision datasets (e.g., CIFAR-10 type sets).
- **Methodology**: Employs standard performance deltas to compute an index score across progressively severe blur and noise corruptions.

## Instructions

### Prerequisites & Dependencies
Ensure you have a recent version of Python installed along with the required libraries. You can install them via `pip`:
```bash
pip install numpy pandas scikit-learn tensorflow opencv-python matplotlib
```

### Running the Bivariate Pipeline (Recommended)
To test the formal state-transition evaluation over a massive real-world dataset (>10M rows), execute the bivariate script. 

**Step 1: Open your terminal or command prompt**
Navigate to the directory where the scripts are located:
```bash
cd "d:\Rabaya_FILES\PersoFiles\THESIS (DONOTTOUCH)\RDI"
```

**Step 2: Run the Prototype Script**
Execute the file using Python:
```bash
python RDI_Prototype_Bivariate.py
```

*Note: The pipeline will automatically fetch the UCI HIGGS dataset (approx. 2.8GB compressed) directly from the remote repository on its first execution. This initial remote data fetch will take a few minutes depending on your internet bandwidth. Successive runs will bypass the download phase and immediately start edge telemetry testing.*

**Step 3: Interpreting the Output**
When you run the script, you will see it advance through two phases:
1. **Phase 1 (Baseline Profiling):** The script reads a massive chunk of data to train the initial classifier and establishes clean baseline limits for accuracy and latency.
2. **Phase 2 (State-Transition Evaluation):** The script begins streaming the remaining frames. You will see real-time console outputs detailing the `Shift Severity` (injected noise), `Acc`, `Lat` (Latency), and the current `NFA State` (0 to 4).

The execution will automatically stop and present the final **Time-To-Failure (TTF)** frame count when the edge node reaches a terminal failure state.

### Running the Image Pipeline
To test the original image degradation codebase:
```bash
python RDI_Prototype1.py
```
*(Ensure you have provided an expected local image dataset directory before running this script).*