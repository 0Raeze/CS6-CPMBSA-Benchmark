import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import copy
import threading
import tracemalloc
import numpy as np
import json

from functions import FUNCTION_SUITE
from bsa import StandardBSA
from cpm_bsa import CPMBSA
from pso import PSO
from rcga import RCGA
from fa import FA


def run_algorithm_with_metrics(AlgoClass, func_info, pop_size, max_evals):
    """Run algorithm, measure runtime (ms) and peak memory (MB)."""
    tracemalloc.start()
    algo = AlgoClass(
        func=func_info["func"],
        bounds=func_info["bounds"],
        dim=func_info["dim"],
        pop_size=pop_size,
        max_evals=max_evals
    )

    start_ns = time.perf_counter_ns()
    result = algo.optimize()
    end_ns = time.perf_counter_ns()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_ms = round((end_ns - start_ns) / 1_000_000.0, 2)
    peak_mb = round(peak / (1024 * 1024), 2)

    if isinstance(result, tuple) and len(result) == 2:
        best_fit, best_pos = result
    else:
        raise RuntimeError("Algorithm.optimize() must return (best_fitness, best_position) tuple.")

    return best_fit, best_pos, runtime_ms, peak_mb


class BenchmarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CS6 - CPMBSA Benchmark Tool")
        self.root.geometry("1200x700")
        self.root.resizable(False, False)

        # Colors
        self.bg = "#2b2b2b"
        self.panel_bg = "#232323"
        self.fg = "#eaeaea"
        self.accent = "#3a7bd5"
        self.text_bg = "#1f1f1f"
        self.text_fg = "#e6e6e6"
        self.root.configure(bg=self.bg)

        # Variables
        self.mode = tk.StringVar(value="standard")
        self.selected_algo1 = tk.StringVar()
        self.selected_algo2 = tk.StringVar()
        self.selected_function = tk.StringVar()
        self.test_size = tk.StringVar(value="medium")
        self.pop_size = tk.StringVar(value="30")
        self.max_evals = tk.StringVar(value="9000")

        # Algorithm mapping
        self.algo_map = {
            "StandardBSA": StandardBSA,
            "CPMBSA": CPMBSA,
            "PSO": PSO,
            "RCGA": RCGA,
            "FA": FA
        }

        self.last_results = []
        self._style_widgets()
        self.build_layout()

    # ---------- Styling ----------
    def _style_widgets(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=self.panel_bg)
        style.configure("TLabel", background=self.panel_bg, foreground=self.fg)
        style.configure("TButton", background=self.accent, foreground=self.fg)
        style.configure("TProgressbar", background=self.accent, troughcolor=self.text_bg)

    # ---------- Layout ----------
    def build_layout(self):
        main = tk.Frame(self.root, bg=self.bg)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Left control panel
        left = tk.Frame(main, bg=self.panel_bg, width=340)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Right output panel (grid layout for equal width)
        right = tk.Frame(main, bg=self.bg)
        right.pack(side="right", fill="both", expand=True)

        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_columnconfigure(1, weight=1)

        tk.Label(left, text="Benchmark Controls", font=("Segoe UI", 16, "bold"),
                 bg=self.panel_bg, fg=self.fg).pack(pady=10)

        # Mode
        mode = tk.LabelFrame(left, text="Mode", bg=self.panel_bg, fg=self.fg)
        mode.pack(fill="x", padx=10, pady=6)
        for text, val in [("Standard Test", "standard"), ("Comparison Mode", "compare")]:
            tk.Radiobutton(mode, text=text, variable=self.mode, value=val,
                           bg=self.panel_bg, fg=self.fg, selectcolor=self.panel_bg,
                           command=self.toggle_mode).pack(anchor="w", padx=6, pady=2)

        # Algorithm selection
        algo_frame = tk.LabelFrame(left, text="Algorithm Selection", bg=self.panel_bg, fg=self.fg)
        algo_frame.pack(fill="x", padx=10, pady=6)
        algo_list = list(self.algo_map.keys())
        tk.Label(algo_frame, text="Algorithm 1:", bg=self.panel_bg, fg=self.fg).pack(anchor="w", padx=6, pady=(4, 2))
        self.algo1_cb = ttk.Combobox(algo_frame, textvariable=self.selected_algo1, values=algo_list, state="readonly")
        self.algo1_cb.pack(padx=6, pady=2)
        self.algo1_cb.bind("<<ComboboxSelected>>", lambda e: self.update_headers())
        tk.Label(algo_frame, text="Algorithm 2:", bg=self.panel_bg, fg=self.fg).pack(anchor="w", padx=6, pady=(6, 2))
        self.algo2_cb = ttk.Combobox(algo_frame, textvariable=self.selected_algo2, values=algo_list, state="readonly")
        self.algo2_cb.pack(padx=6, pady=2)
        self.algo2_cb.bind("<<ComboboxSelected>>", lambda e: self.update_headers())

        # Function selection
        func = tk.LabelFrame(left, text="Benchmark Function", bg=self.panel_bg, fg=self.fg)
        func.pack(fill="x", padx=10, pady=6)
        self.func_cb = ttk.Combobox(func, textvariable=self.selected_function,
                                    values=list(FUNCTION_SUITE.keys()), state="readonly", width=28)
        self.func_cb.pack(padx=6, pady=6)

        # Test size
        size = tk.LabelFrame(left, text="Test Size", bg=self.panel_bg, fg=self.fg)
        size.pack(fill="x", padx=10, pady=6)
        self.size_cb = ttk.Combobox(size, textvariable=self.test_size,
                                    values=["small", "medium", "large"], state="readonly", width=20)
        self.size_cb.pack(padx=6, pady=6)

        # Parameters
        params = tk.LabelFrame(left, text="Algorithm Parameters", bg=self.panel_bg, fg=self.fg)
        params.pack(fill="x", padx=10, pady=6)
        for label, var in [("Population size:", self.pop_size), ("Max evaluations:", self.max_evals)]:
            row = tk.Frame(params, bg=self.panel_bg)
            row.pack(fill="x", padx=6, pady=3)
            tk.Label(row, text=label, bg=self.panel_bg, fg=self.fg).pack(side="left")
            tk.Entry(row, textvariable=var, width=8, bg=self.text_bg, fg=self.text_fg,
                     insertbackground=self.text_fg).pack(side="right")

        # Buttons
        bframe = tk.Frame(left, bg=self.panel_bg)
        bframe.pack(fill="x", padx=10, pady=10)
        self.run_btn = tk.Button(bframe, text="Run", bg=self.accent, fg=self.fg, command=self.start_run_thread)
        self.run_btn.pack(fill="x", pady=(0, 6))
        tk.Button(bframe, text="Notes", bg="#444", fg=self.fg, command=self.show_notes).pack(fill="x", pady=(0, 6))
        tk.Button(bframe, text="Download Results (.xlsx)", bg="#4a4a4a", fg=self.fg,
                  command=self.download_results).pack(fill="x")

        # Progress bar
        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(6, 0))

        # --- Right side outputs (symmetrical grid) ---
        self.out1_frame = tk.Frame(right, bg=self.panel_bg)
        self.out2_frame = tk.Frame(right, bg=self.panel_bg)

        self.out1_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=6)
        self.header1 = tk.Label(self.out1_frame, text="Algorithm 1", bg=self.panel_bg, fg=self.fg,
                                font=("Segoe UI", 11, "bold"))
        self.header1.pack(anchor="nw", padx=8, pady=(8, 4))
        self.text1 = tk.Text(self.out1_frame, bg=self.text_bg, fg=self.text_fg, wrap="none")
        self.text1.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.header2 = tk.Label(self.out2_frame, text="Algorithm 2", bg=self.panel_bg, fg=self.fg,
                                font=("Segoe UI", 11, "bold"))
        self.header2.pack(anchor="nw", padx=8, pady=(8, 4))
        self.text2 = tk.Text(self.out2_frame, bg=self.text_bg, fg=self.text_fg, wrap="none")
        self.text2.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.text1.insert("end", "Output 1...\n")
        self.text2.insert("end", "Output 2...\n")

        self.toggle_mode()

    # ---------- UI Handlers ----------
    def toggle_mode(self):
        if self.mode.get() == "compare":
            self.out2_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=6)
            self.algo2_cb.configure(state="readonly")
        else:
            self.out2_frame.grid_forget()
            self.algo2_cb.configure(state="disabled")
        self.update_headers()

    def update_headers(self):
        self.header1.config(text=self.selected_algo1.get() or "Algorithm 1")
        self.header2.config(text=self.selected_algo2.get() or "Algorithm 2")

    def show_notes(self):
        msg = (
            "Notes:\n"
            "- SixHumpCamel is fixed at dim=2.\n"
            "- Threaded execution keeps GUI responsive.\n"
            "- Memory usage measured via tracemalloc (peak MB).\n"
            "- Excel export shows runtime & memory vs input size."
        )
        messagebox.showinfo("Notes", msg)

    # ---------- Threaded Run ----------
    def start_run_thread(self):
        t = threading.Thread(target=self.run_test, daemon=True)
        self.run_btn.config(state="disabled")
        self.progress.start(10)
        t.start()

    def run_test(self):
        try:
            self._run_test_core()
        finally:
            self.progress.stop()
            self.run_btn.config(state="normal")

    # ---------- Core Logic ----------
    def _run_test_core(self):
        mode = self.mode.get()
        func_key = self.selected_function.get()
        algo1 = self.selected_algo1.get()
        algo2 = self.selected_algo2.get()
        size = self.test_size.get()

        if not func_key or not algo1:
            messagebox.showwarning("Missing Input", "Select function and Algorithm 1.")
            return
        if mode == "compare" and not algo2:
            messagebox.showwarning("Missing Input", "Select Algorithm 2 for comparison.")
            return

        try:
            pop = int(self.pop_size.get())
            evals = int(self.max_evals.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Population and max evals must be integers.")
            return

        func = copy.deepcopy(FUNCTION_SUITE[func_key])
        if func_key == "F52_SixHumpCamel":
            func["dim"] = 2
        else:
            func["dim"] = {"small": 10, "medium": 40, "large": 100}.get(size, 40)

        self.text1.delete(1.0, "end")
        self.text2.delete(1.0, "end")
        header = f"Function: {func_key}\nDim: {func['dim']}  Size: {size}\nPop: {pop}  Evals: {evals}\n\n"
        self.last_results = []

        if mode == "standard":
            Algo = self.algo_map[algo1]
            self.text1.insert("end", header)
            try:
                fit, pos, rt, mem = run_algorithm_with_metrics(Algo, func, pop, evals)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return

            self.text1.insert("end", f"{algo1}\nBestFitness: {fit:.6e}\n"
                                     f"Time: {rt:.2f} ms\nPeakMem: {mem:.2f} MB\nPos: {np.round(pos, 5)}\n")

            self.last_results.append({
                "Algorithm": algo1, "Function": func_key, "InputSize": size, "Dimension": func["dim"],
                "Runtime_ms": rt, "Memory_MB": mem, "BestFitness": fit, "PopSize": pop, "MaxEvals": evals
            })

        else:
            Algo1 = self.algo_map[algo1]
            Algo2 = self.algo_map[algo2]
            self.text1.insert("end", header)
            self.text2.insert("end", header)
            fit1, pos1, rt1, mem1 = run_algorithm_with_metrics(Algo1, func, pop, evals)
            fit2, pos2, rt2, mem2 = run_algorithm_with_metrics(Algo2, func, pop, evals)

            self.text1.insert("end", f"{algo1}\nBestFitness: {fit1:.6e}\nTime: {rt1:.2f} ms\nPeakMem: {mem1:.2f} MB\n")
            self.text2.insert("end", f"{algo2}\nBestFitness: {fit2:.6e}\nTime: {rt2:.2f} ms\nPeakMem: {mem2:.2f} MB\n")

            better = algo1 if fit1 < fit2 else algo2
            summary = f"\nBetter performer (lower fitness): {better}\n"
            self.text1.insert("end", summary)
            self.text2.insert("end", summary)

            self.last_results.extend([
                {"Algorithm": algo1, "Function": func_key, "InputSize": size, "Dimension": func["dim"],
                 "Runtime_ms": rt1, "Memory_MB": mem1, "BestFitness": fit1, "PopSize": pop, "MaxEvals": evals},
                {"Algorithm": algo2, "Function": func_key, "InputSize": size, "Dimension": func["dim"],
                 "Runtime_ms": rt2, "Memory_MB": mem2, "BestFitness": fit2, "PopSize": pop, "MaxEvals": evals}
            ])

    # ---------- Excel Export ----------
    def download_results(self):
        if not self.last_results:
            messagebox.showwarning("No Data", "Run a test first.")
            return
        try:
            import pandas as pd
        except Exception:
            messagebox.showerror("Missing pandas", "Install with:\n pip install pandas openpyxl")
            return

        df = pd.DataFrame(self.last_results)
        file = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel Workbook", "*.xlsx")],
                                            title="Save results as...")
        if not file:
            return
        try:
            df.to_excel(file, index=False)
            messagebox.showinfo("Saved", f"Results saved:\n{file}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = BenchmarkGUI(root)
    root.mainloop()