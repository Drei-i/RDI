import os
import time
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================================================================
# 1. THE RDI THEORETICAL FRAMEWORK (Model-Agnostic)
# ==============================================================================
# This section represents the universal testing methodology. It has zero knowledge 
# of what machine learning model you are using or how it was trained.

class BivariateRDI_DFA:
    """
    5-State Deterministic Finite Automaton tracking Accuracy & Latency.
    State 0: Optimal Operation
    State 1: Minor Degradation
    State 2: Critical Degradation
    State 3: Terminal Failure (qf - either accuracy or latency fails)
    State 4: Fatal Lockup (q_fatal - both fail simultaneously)
    """
    def __init__(self, acc_baseline, lat_baseline):
        self.acc_baseline = acc_baseline
        self.lat_baseline = lat_baseline
        
        # Algorithmic Evaluation Thresholds mapped to degradation severity
        self.acc_minor = acc_baseline * 0.90
        self.acc_critical = acc_baseline * 0.70
        self.acc_terminal = acc_baseline * 0.50
        
        self.lat_minor = lat_baseline * 1.5
        self.lat_critical = lat_baseline * 3.0
        self.lat_terminal = lat_baseline * 5.0
        
        self.current_state = 0
        self.ttf = 0  # Time-to-Failure in discrete frames
        
    def evaluate_frame(self, accuracy, latency):
        # 1. Check for absorbing final states
        acc_fail = accuracy < self.acc_terminal
        lat_fail = latency > self.lat_terminal
        
        if acc_fail and lat_fail:
            self.current_state = 4 # q_fatal
        elif acc_fail or lat_fail:
            self.current_state = 3 # qf
            
        # 2. Check operational states (Reversibility allowed)
        else:
            if accuracy < self.acc_critical or latency > self.lat_critical:
                self.current_state = 2
            elif accuracy < self.acc_minor or latency > self.lat_minor:
                self.current_state = 1
            else:
                self.current_state = 0
                
        # 3. Update TTF if operational
        if self.current_state < 3:
            self.ttf += 1
            
        return self.current_state

def plot_bivariate_ui(metrics, dfa):
    frames = [m['frame'] for m in metrics]
    accuracies = [m['acc'] for m in metrics]
    latencies = [m['lat'] for m in metrics]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.title(f'Bivariate RDI Degradation Curve (TTF: {dfa.ttf} Frames)')
    
    color_acc = 'tab:blue'
    ax1.set_xlabel('Inference Frames Processed (Time)')
    ax1.set_ylabel('Predictive Accuracy', color=color_acc)
    ax1.plot(frames, accuracies, color=color_acc, marker='o', markersize=3, label='Actual Accuracy')
    ax1.axhline(y=dfa.acc_terminal, color=color_acc, linestyle='--', alpha=0.5, label='Terminal Accuracy Limit')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    
    ax2 = ax1.twinx()
    color_lat = 'tab:red'
    ax2.set_ylabel('Inference Latency (seconds/row)', color=color_lat)
    ax2.plot(frames, latencies, color=color_lat, marker='x', markersize=3, label='Actual Latency')
    ax2.axhline(y=dfa.lat_terminal, color=color_lat, linestyle='--', alpha=0.5, label='Terminal Latency Limit')
    ax2.tick_params(axis='y', labelcolor=color_lat)
    
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    ax1.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    plt.show()

class UniversalRDIEvaluator:
    """
    The Universal Testing Framework. 
    Accepts ANY model prediction function and evaluates its degradation.
    """
    def __init__(self, predict_func, baseline_acc, baseline_lat):
        self.predict_func = predict_func # The user's external prediction function
        self.dfa = BivariateRDI_DFA(baseline_acc, baseline_lat)
        self.metrics_log = []
        
    def evaluate_frame(self, X_frame, y_frame, noise_level, frame_idx):
        # 1. Apply algorithmic corruption simulating distribution shift
        noise = np.random.normal(0, noise_level, X_frame.shape)
        X_degraded = X_frame + noise
        
        # 2. Inference Simulation using the provided AGNOSTIC model function
        start_inf = time.time()
        preds = self.predict_func(X_degraded)
        
        # 3. Artificial penalty simulating physical CPU stress on edge node
        # Lowered multiplier to 0.0002 to stretch the degradation curve across all remaining data
        simulated_delay = noise_level * 0.0002 
        time.sleep(simulated_delay)
        
        # 4. Calculate Bivariate Metrics
        latency = (time.time() - start_inf) / len(X_frame)
        acc = accuracy_score(y_frame, preds)
        
        # 5. Evaluate state through the DFA
        state = self.dfa.evaluate_frame(acc, latency)
        self.metrics_log.append({'frame': frame_idx, 'acc': acc, 'lat': latency})
        
        print(f"Frame {frame_idx:03d} | Shift: {noise_level:.2f} | Acc: {acc:.4f} | Lat: {latency:.8f}s | DFA State: {state}")
        
        # 6. Check for terminal failure
        if state >= 3:
            self._print_diagnostic_report(noise_level, acc, latency, state)
            return True # True means terminal failure reached
            
        return False
        
    def _print_diagnostic_report(self, noise, acc, lat, state):
        print("\n" + "="*50)
        print(" [!] TERMINAL STATE REACHED [!] ")
        print("="*50)
        if state == 3:
            print("Failure Type : [qf] Single Metric Failure")
        else:
            print("Failure Type : [q_fatal] Simultaneous Hardware/Software Lockup")
        
        print(f"\n--- DEGRADATION DIAGNOSTIC REPORT ---")
        print(f"Final Shift Severity : {noise:.2f}")
        print(f"Final Accuracy       : {acc:.4f}  (Terminal Limit: < {self.dfa.acc_terminal:.4f})")
        print(f"Final Latency        : {lat:.8f}s (Terminal Limit: > {self.dfa.lat_terminal:.8f}s)")
        
        print("\n--- FAILURE CAUSE ---")
        if lat > self.dfa.lat_terminal:
            print("[X] LATENCY CRASH: CPU thermal throttling threshold breached.")
        if acc < self.dfa.acc_terminal:
            print("[X] ACCURACY CRASH: Feature extraction failed.")
            
        print("="*50)
        print(f"[*] FINAL TIME-TO-FAILURE (TTF): {self.dfa.ttf} frames successfully processed.")
        print("="*50)
        
        # Launch UI Visualization
        print("Launching Matplotlib Visualization UI...")
        plot_bivariate_ui(self.metrics_log, self.dfa)


# ==============================================================================
# 2. USER PAYLOAD SCRIPT (How a developer uses the RDI Framework)
# ==============================================================================
# This section represents what a developer would write. They load their own data,
# build their own model in whatever library they want, and simply plug it into 
# the UniversalRDIEvaluator above.

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
DATA_FILE = "HIGGS.csv.gz"

def ensure_dataset():
    if not os.path.exists(DATA_FILE):
        print(f"Dataset not found. Downloading 11 Million Rows from UCI (HIGGS.csv.gz) (~2.8GB)...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download complete.")
    else:
        print(f"Dataset {DATA_FILE} already exists locally.")



def user_testing_pipeline(models_to_test=None):
    ensure_dataset()
    
    # ---------------------------------------------------------
    # STEP 1: INITIALIZE YOUR MODEL DICTIONARY (EASY INDEXING)
    # ---------------------------------------------------------
    if models_to_test is None:
        models_to_test = {
            "Linear_SGD_LogLoss": SGDClassifier(loss='log_loss', max_iter=1, random_state=42),
            "Linear_SGD_Hinge": SGDClassifier(loss='hinge', max_iter=1, random_state=42),
            "Deep_Neural_Net": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42)
        }
    
    # We lowered baseline to 2M rows so testing 3 models doesn't take all day.
    baseline_rows = 2000000
    train_chunk = 1000000 
    
    scoreboard = {}
    
    for model_name, my_model in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"[*] NOW EVALUATING MODEL: {model_name}")
        print(f"{'='*60}")
        
        scaler = StandardScaler()
        
        # ---------------------------------------------------------
        # STEP 2: TRAIN YOUR MODEL AND ESTABLISH BASELINE
        # ---------------------------------------------------------
        print(f"\n--- BASELINE PROFILING ({model_name}) ---")
        rows_processed = 0
        
        data_iterator = pd.read_csv(DATA_FILE, chunksize=train_chunk, nrows=baseline_rows, header=None)
        
        for chunk in data_iterator:
            y_train = chunk.iloc[:, 0].values
            X_train = chunk.iloc[:, 1:].values
            
            scaler.partial_fit(X_train)
            X_scaled = scaler.transform(X_train)
            my_model.partial_fit(X_scaled, y_train, classes=[0, 1])
            
            rows_processed += len(chunk)
            print(f"  -> Trained on {rows_processed:,} / {baseline_rows:,} rows")
            
        eval_size = 5000
        X_eval = X_scaled[:eval_size]
        y_eval = y_train[:eval_size]
        
        start_time = time.time()
        base_preds = my_model.predict(X_eval)
        baseline_latency = (time.time() - start_time) / eval_size
        baseline_acc = accuracy_score(y_eval, base_preds)
        
        print(f"Baseline Accuracy: {baseline_acc:.4f}")
        print(f"Baseline Latency per row: {baseline_latency:.8f} seconds")
        
        # ---------------------------------------------------------
        # STEP 3: CREATE A PREDICTION WRAPPER
        # ---------------------------------------------------------
        def my_prediction_wrapper(X_raw, model_ref=my_model, scaler_ref=scaler):
            X_preprocessed = scaler_ref.transform(X_raw)
            return model_ref.predict(X_preprocessed)
            
        # ---------------------------------------------------------
        # STEP 4: PLUG INTO THE UNIVERSAL RDI EVALUATOR
        # ---------------------------------------------------------
        rdi_tester = UniversalRDIEvaluator(predict_func=my_prediction_wrapper, 
                                           baseline_acc=baseline_acc, 
                                           baseline_lat=baseline_latency)
                                           
        # ---------------------------------------------------------
        # STEP 5: STREAM DATA THROUGH THE TESTER
        # ---------------------------------------------------------
        print(f"\n--- PHASE 2: STATE-TRANSITION EVALUATION ({model_name}) ---")
        chunk_size = 5000
        noise_level = 0.0
        frame_idx = 1
        
        phase2_iterator = pd.read_csv(DATA_FILE, chunksize=chunk_size, skiprows=baseline_rows, header=None)
        
        for chunk in phase2_iterator:
            y_test = chunk.iloc[:, 0].values
            X_test = chunk.iloc[:, 1:].values
            
            is_failed = rdi_tester.evaluate_frame(X_frame=X_test, 
                                                  y_frame=y_test, 
                                                  noise_level=noise_level, 
                                                  frame_idx=frame_idx)
                                                  
            if is_failed:
                break
                
            noise_level += 0.15
            frame_idx += 1
            
        # Save score for final comparison
        scoreboard[model_name] = rdi_tester.dfa.ttf
        
    # ---------------------------------------------------------
    # FINAL RDI BENCHMARK SCOREBOARD
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("   FINAL RDI DEGRADATION SCOREBOARD   ")
    print("="*40)
    for m_name, ttf in scoreboard.items():
        print(f"Model: {m_name.ljust(20)} | Time-To-Failure (TTF): {ttf} frames")
    print("="*60)

if __name__ == "__main__":
    user_testing_pipeline()
