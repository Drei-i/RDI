import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import io

# Import machine learning dependencies
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# Import your Universal RDI Framework!
from RDI_Prototype_Bivariate import user_testing_pipeline

class RedirectText(io.StringIO):
    """
    Safely redirects Python's standard print() statements into the Tkinter Text widget 
    even when running from a background thread.
    """
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        super().__init__()

    def write(self, string):
        self.root.after(0, self._insert_text, string)

    def _insert_text(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        
    def flush(self):
        pass

class RDIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RDI Degradation Benchmark Suite")
        self.root.geometry("900x650")
        
        # Apply a slightly cleaner theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # --- HEADER ---
        header_frame = ttk.Frame(root, padding=10)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="RDI Universal Model Benchmarker", font=("Arial", 16, "bold")).pack()
        ttk.Label(header_frame, text="Select the machine learning models you wish to evaluate against the edge degradation pipeline.", font=("Arial", 10)).pack()
        
        # --- MODEL SELECTION ---
        model_frame = ttk.LabelFrame(root, text="Index / Select Models", padding=10)
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.var_sgd = tk.BooleanVar(value=True)
        self.var_hinge = tk.BooleanVar(value=False)
        self.var_mlp = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(model_frame, text="Linear SGD (Log Loss) - Lightweight", variable=self.var_sgd).grid(row=0, column=0, padx=10, sticky="w")
        ttk.Checkbutton(model_frame, text="Linear SGD (Hinge Loss) - Aggressive", variable=self.var_hinge).grid(row=0, column=1, padx=10, sticky="w")
        ttk.Checkbutton(model_frame, text="Deep Neural Network (MLP) - Heavyweight", variable=self.var_mlp).grid(row=0, column=2, padx=10, sticky="w")
        
        # --- START BUTTON ---
        self.run_btn = ttk.Button(root, text="START BENCHMARK SUITE", command=self.start_benchmark)
        self.run_btn.pack(pady=10)
        
        # --- TELEMETRY CONSOLE ---
        console_frame = ttk.LabelFrame(root, text="Live RDI Telemetry", padding=10)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Dark hacker-style terminal text area
        self.text_area = tk.Text(console_frame, bg="#1E1E1E", fg="#00FF41", font=("Consolas", 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Redirect stdout so all print() calls from RDI_Prototype_Bivariate show up in the UI!
        sys.stdout = RedirectText(self.text_area, self.root)
        
    def start_benchmark(self):
        # Build the dynamic dictionary based on what the user checked in the UI!
        models_to_test = {}
        if self.var_sgd.get():
            models_to_test["Linear_SGD_LogLoss"] = SGDClassifier(loss='log_loss', max_iter=1, random_state=42)
        if self.var_hinge.get():
            models_to_test["Linear_SGD_Hinge"] = SGDClassifier(loss='hinge', max_iter=1, random_state=42)
        if self.var_mlp.get():
            models_to_test["Deep_Neural_Net"] = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42)
            
        if not models_to_test:
            messagebox.showwarning("Warning", "Please select at least one model to index.")
            return
            
        self.run_btn.config(state=tk.DISABLED)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Initializing Universal RDI Benchmark Suite...\n")
        
        # Run pipeline in a background thread so the UI doesn't freeze during 10M row training
        threading.Thread(target=self.run_pipeline_thread, args=(models_to_test,), daemon=True).start()
        
    def run_pipeline_thread(self, models_to_test):
        try:
            # We literally just pass the dictionary into your agnostic framework!
            user_testing_pipeline(models_to_test)
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("Complete", "Benchmarking Complete! See the final scoreboard."))

if __name__ == "__main__":
    root = tk.Tk()
    app = RDIApp(root)
    root.mainloop()
