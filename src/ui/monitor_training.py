#!/usr/bin/env python
"""
Progressive Training Monitoring Dashboard

This script provides a real-time dashboard for monitoring progressive training,
showing visualizations of training progress, knowledge transfer, and system metrics.
"""

import os
import sys
import json
import time
import argparse
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Add project root to system path to ensure imports work
sys.path.insert(0, project_root)

try:
    from src.utils.training_visualizer import ProgressiveTrainingVisualizer
except ImportError:
    print("Warning: Could not import ProgressiveTrainingVisualizer.")

# Directory constants
MODELS_DIR_DEFAULT = os.path.join(project_root, "Models")
CONFIG_FILE = os.path.join(project_root, "configs", "config.json")

class TrainingMonitor:
    """
    GUI for monitoring progressive training.
    
    This class provides a Tkinter-based dashboard for real-time monitoring of
    training progress, knowledge transfer events, and system metrics.
    """
    
    def __init__(self, models_dir=MODELS_DIR_DEFAULT, update_interval=5):
        """
        Initialize the training monitor.
        
        Args:
            models_dir: Directory containing model folders
            update_interval: Data refresh interval in seconds
        """
        self.models_dir = models_dir
        self.update_interval = update_interval
        
        # Initialize visualizer
        self.visualizer = ProgressiveTrainingVisualizer(
            output_dir=os.path.join(models_dir, "monitoring")
        )
        
        # Data storage
        self.training_history = {}
        self.transfer_history = []
        self.memory_usage = {}
        
        # Tracking variables
        self.active_buckets = []
        self.current_bucket = None
        self.watching = False
        self.last_update_time = 0
        
        # Initialize UI
        self._create_ui()
        
        # Start with a data refresh
        self._update_data()
    
    def _create_ui(self):
        """Create the tkinter UI."""
        # Create root window
        self.root = tk.Tk()
        self.root.title("Progressive Training Monitor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel (top)
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Status indicators
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.status_label = ttk.Label(status_frame, text="Idle", foreground="gray")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_frame, text="Active Bucket:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.bucket_label = ttk.Label(status_frame, text="None", foreground="gray")
        self.bucket_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(status_frame, text="Last Update:").grid(row=2, column=0, sticky=tk.W, padx=5)
        self.update_label = ttk.Label(status_frame, text="Never", foreground="gray")
        self.update_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.watch_button = ttk.Button(button_frame, text="Start Watching", 
                                       command=self._toggle_watching)
        self.watch_button.pack(side=tk.LEFT, padx=5)
        
        self.refresh_button = ttk.Button(button_frame, text="Refresh Now", 
                                        command=self._update_data)
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        
        self.report_button = ttk.Button(button_frame, text="Generate Report", 
                                       command=self._generate_report)
        self.report_button.pack(side=tk.LEFT, padx=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Dashboard tab
        self.dashboard_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dashboard_tab, text="Dashboard")
        
        # Training progress tab
        self.progress_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.progress_tab, text="Training Progress")
        
        # Knowledge transfer tab
        self.transfer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.transfer_tab, text="Knowledge Transfer")
        
        # System metrics tab
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="System Metrics")
        
        # Log tab
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Training Log")
        
        # Create dashboard content
        self._create_dashboard_tab()
        self._create_progress_tab()
        self._create_transfer_tab()
        self._create_metrics_tab()
        self._create_log_tab()
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_dashboard_tab(self):
        """Create content for the dashboard tab."""
        # Create frame for the matplotlib figure
        self.dashboard_fig_frame = ttk.Frame(self.dashboard_tab)
        self.dashboard_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial empty figure
        self.dashboard_fig = plt.Figure(figsize=(10, 8))
        self.dashboard_canvas = FigureCanvasTkAgg(self.dashboard_fig, master=self.dashboard_fig_frame)
        self.dashboard_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_progress_tab(self):
        """Create content for the training progress tab."""
        # Create frame for the matplotlib figure
        self.progress_fig_frame = ttk.Frame(self.progress_tab)
        self.progress_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings frame
        settings_frame = ttk.Frame(self.progress_tab)
        settings_frame.pack(fill=tk.X)
        
        ttk.Label(settings_frame, text="Metric:").pack(side=tk.LEFT, padx=5)
        self.metric_var = tk.StringVar(value="reward")
        metric_options = ttk.Combobox(settings_frame, textvariable=self.metric_var, 
                                      values=["reward", "loss", "win_rate", "profit_factor"])
        metric_options.pack(side=tk.LEFT, padx=5)
        metric_options.bind("<<ComboboxSelected>>", self._update_progress_plot)
        
        ttk.Label(settings_frame, text="Smoothing:").pack(side=tk.LEFT, padx=5)
        self.smooth_var = tk.IntVar(value=5)
        smooth_options = ttk.Combobox(settings_frame, textvariable=self.smooth_var, 
                                     values=[1, 3, 5, 10, 15])
        smooth_options.pack(side=tk.LEFT, padx=5)
        smooth_options.bind("<<ComboboxSelected>>", self._update_progress_plot)
        
        # Initial empty figure
        self.progress_fig = plt.Figure(figsize=(10, 6))
        self.progress_canvas = FigureCanvasTkAgg(self.progress_fig, master=self.progress_fig_frame)
        self.progress_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_transfer_tab(self):
        """Create content for the knowledge transfer tab."""
        # Create frame for the matplotlib figure
        self.transfer_fig_frame = ttk.Frame(self.transfer_tab)
        self.transfer_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial empty figure
        self.transfer_fig = plt.Figure(figsize=(10, 6))
        self.transfer_canvas = FigureCanvasTkAgg(self.transfer_fig, master=self.transfer_fig_frame)
        self.transfer_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Transfer details list
        details_frame = ttk.LabelFrame(self.transfer_tab, text="Transfer Details")
        details_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview widget
        columns = ("Episode", "Source", "Target", "Types", "Success")
        self.transfer_tree = ttk.Treeview(details_frame, columns=columns, show="headings")
        
        # Define headings
        for col in columns:
            self.transfer_tree.heading(col, text=col)
            self.transfer_tree.column(col, width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.transfer_tree.yview)
        self.transfer_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.transfer_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_metrics_tab(self):
        """Create content for the system metrics tab."""
        # Create frame for the matplotlib figure
        self.metrics_fig_frame = ttk.Frame(self.metrics_tab)
        self.metrics_fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial empty figure
        self.metrics_fig = plt.Figure(figsize=(10, 6))
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=self.metrics_fig_frame)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_log_tab(self):
        """Create content for the log tab."""
        # Log selection frame
        select_frame = ttk.Frame(self.log_tab)
        select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(select_frame, text="Bucket:").pack(side=tk.LEFT, padx=5)
        self.log_bucket_var = tk.StringVar(value="All")
        self.log_bucket_combo = ttk.Combobox(select_frame, textvariable=self.log_bucket_var)
        self.log_bucket_combo.pack(side=tk.LEFT, padx=5)
        self.log_bucket_combo.bind("<<ComboboxSelected>>", self._update_log)
        
        refresh_button = ttk.Button(select_frame, text="Refresh Log", command=self._update_log)
        refresh_button.pack(side=tk.RIGHT, padx=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(self.log_tab, wrap=tk.WORD, height=30)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _toggle_watching(self):
        """Toggle the automatic data refresh."""
        if self.watching:
            self.watching = False
            self.watch_button.config(text="Start Watching")
        else:
            self.watching = True
            self.watch_button.config(text="Stop Watching")
            # Start the refresh thread
            self.refresh_thread = threading.Thread(target=self._watch_thread, daemon=True)
            self.refresh_thread.start()
    
    def _watch_thread(self):
        """Background thread for automatic data refresh."""
        while self.watching:
            self._update_data()
            time.sleep(self.update_interval)
    
    def _generate_report(self):
        """Generate a comprehensive training report."""
        # Check if we have data
        if not self.training_history or not hasattr(self, 'visualizer'):
            tk.messagebox.showinfo("Report Generation", "No training data available to generate a report.")
            return
        
        # Generate report using visualizer
        try:
            # Create timestamp for report folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(self.models_dir, "reports", f"training_report_{timestamp}")
            os.makedirs(os.path.dirname(report_dir), exist_ok=True)
            
            # Generate report
            report_path = self.visualizer.generate_training_report(
                self.training_history,
                self.transfer_history,
                output_path=report_dir
            )
            
            # Inform user
            self.status_bar.config(text=f"Report generated at: {report_path}")
            tk.messagebox.showinfo("Report Generation", 
                                  f"Report generated successfully!\n\nSaved to: {report_path}")
            
            # Try to open the report
            try:
                os.startfile(report_path)  # Windows
            except AttributeError:
                try:
                    import subprocess
                    subprocess.call(['open', report_path])  # macOS
                except:
                    pass  # Failed to open, but that's okay
        except Exception as e:
            tk.messagebox.showerror("Report Generation Error", f"Failed to generate report: {str(e)}")
            self.status_bar.config(text="Error generating report.")
    
    def _update_data(self):
        """Refresh monitoring data from disk."""
        self.status_bar.config(text="Updating data...")
        
        # Find active buckets
        self._find_active_buckets()
        
        # Get current active bucket
        self.current_bucket = self._get_current_bucket()
        if self.current_bucket:
            self.bucket_label.config(text=self.current_bucket, foreground="blue")
        else:
            self.bucket_label.config(text="None", foreground="gray")
        
        # Update training history
        for bucket in self.active_buckets:
            self._load_training_history(bucket)
        
        # Update transfer history
        self._load_transfer_history()
        
        # Update memory usage
        self._load_memory_usage()
        
        # Update visualizations
        self._update_dashboard()
        self._update_progress_plot()
        self._update_transfer_plot()
        self._update_metrics_plot()
        self._update_log()
        
        # Update status
        self.last_update_time = time.time()
        self.update_label.config(text=datetime.now().strftime("%H:%M:%S"), foreground="blue")
        self.status_bar.config(text=f"Data updated. Active buckets: {', '.join(self.active_buckets) if self.active_buckets else 'None'}")
        
        # Handle status text
        if self.current_bucket:
            self.status_label.config(text="Training Active", foreground="green")
        elif self.active_buckets:
            self.status_label.config(text="Trained", foreground="blue")
        else:
            self.status_label.config(text="Idle", foreground="gray")
    
    def _find_active_buckets(self):
        """Find all active bucket directories."""
        buckets = []
        try:
            for item in os.listdir(self.models_dir):
                if os.path.isdir(os.path.join(self.models_dir, item)):
                    if item in ["Scalping", "Short", "Medium", "Long"]:
                        buckets.append(item)
        except Exception as e:
            print(f"Error finding buckets: {e}")
        self.active_buckets = buckets
    
    def _get_current_bucket(self):
        """Determine which bucket is currently being trained."""
        for bucket in self.active_buckets:
            recovery_file = os.path.join(self.models_dir, bucket, "recovery_state.json")
            if os.path.exists(recovery_file):
                try:
                    with open(recovery_file, "r") as f:
                        data = json.load(f)
                        if data.get("is_training", False):
                            return bucket
                except:
                    pass
        return None
    
    def _load_training_history(self, bucket):
        """Load training history for a specific bucket."""
        history_file = os.path.join(self.models_dir, bucket, "training_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    return json.load(f)
            except:
                pass
        
        # Try to get performance data from the progress file
        progress_file = os.path.join(self.models_dir, bucket, "progress.csv")
        if os.path.exists(progress_file):
            try:
                import pandas as pd
                df = pd.read_csv(progress_file)
                
                # Convert DataFrame to list of dictionaries
                history = []
                for _, row in df.iterrows():
                    entry = row.to_dict()
                    history.append(entry)
                
                return history
            except:
                pass
        
        return None
    
    def _load_transfer_history(self):
        """Load knowledge transfer history."""
        transfer_file = os.path.join(self.models_dir, "knowledge_transfer", "transfer_history.json")
        if os.path.exists(transfer_file):
            try:
                with open(transfer_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _load_memory_usage(self):
        """Load memory usage history."""
        memory_file = os.path.join(self.models_dir, "memory_usage.json")
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _update_dashboard(self):
        """Update the dashboard visualization."""
        # Clear previous figure
        self.dashboard_fig.clear()
        
        # Create dashboard
        if hasattr(self, 'visualizer') and self.training_history:
            # Extract memory usage in the correct format
            memory_data = None
            if self.memory_usage:
                episodes = []
                usage_values = []
                
                for bucket, usage in self.memory_usage.items():
                    for ep, val in usage.items():
                        episodes.append(int(ep))
                        usage_values.append(val)
                
                if episodes and usage_values:
                    # Sort by episode
                    sorted_pairs = sorted(zip(episodes, usage_values))
                    episodes = [pair[0] for pair in sorted_pairs]
                    usage_values = [pair[1] for pair in sorted_pairs]
                    
                    memory_data = {
                        'episodes': episodes,
                        'usage': usage_values
                    }
            
            # Use the visualizer to create the dashboard
            try:
                # Replace the figure with the one created by the visualizer
                plt.close(self.dashboard_fig)
                self.dashboard_fig = self.visualizer.plot_training_dashboard(
                    self.training_history,
                    self.transfer_history,
                    memory_data,
                    title="Progressive Training Dashboard"
                )
                
                # Update canvas
                self.dashboard_canvas.figure = self.dashboard_fig
                self.dashboard_canvas.draw()
            except Exception as e:
                print(f"Error updating dashboard: {e}")
                # Create a simple message if visualization fails
                ax = self.dashboard_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Could not generate dashboard visualization\nError: {str(e)}", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                self.dashboard_canvas.draw()
        else:
            # Create a "no data" message
            ax = self.dashboard_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No training data available", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.dashboard_canvas.draw()
    
    def _update_progress_plot(self, event=None):
        """Update the training progress visualization."""
        # Clear previous figure
        self.progress_fig.clear()
        
        # Get selected metric and smoothing
        metric = self.metric_var.get()
        smooth = int(self.smooth_var.get())
        
        # Create plot if we have data
        if hasattr(self, 'visualizer') and self.training_history:
            try:
                # Replace the figure with the one created by the visualizer
                plt.close(self.progress_fig)
                self.progress_fig = self.visualizer.plot_training_progress(
                    self.training_history,
                    metrics=[metric],
                    smooth_window=smooth,
                    title=f"{metric.capitalize()} Progress"
                )
                
                # Update canvas
                self.progress_canvas.figure = self.progress_fig
                self.progress_canvas.draw()
            except Exception as e:
                print(f"Error updating progress plot: {e}")
                # Create a simple message if visualization fails
                ax = self.progress_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Could not generate progress visualization\nError: {str(e)}", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                self.progress_canvas.draw()
        else:
            # Create a "no data" message
            ax = self.progress_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No training data available", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.progress_canvas.draw()
    
    def _update_transfer_plot(self):
        """Update the knowledge transfer visualization."""
        # Clear previous figure
        self.transfer_fig.clear()
        
        # Create plot if we have data
        if hasattr(self, 'visualizer') and self.transfer_history:
            try:
                # Replace the figure with the one created by the visualizer
                plt.close(self.transfer_fig)
                self.transfer_fig = self.visualizer.plot_knowledge_transfer(
                    self.transfer_history,
                    buckets=list(self.training_history.keys()),
                    title="Knowledge Transfer Events"
                )
                
                # Update canvas
                self.transfer_canvas.figure = self.transfer_fig
                self.transfer_canvas.draw()
                
                # Update transfer details list
                self.transfer_tree.delete(*self.transfer_tree.get_children())
                for entry in self.transfer_history:
                    episode = entry.get('episode', 0)
                    source = entry.get('source', 'Unknown')
                    target = entry.get('target', 'Unknown')
                    types = ', '.join(entry.get('transfer_types', []))
                    success = "Yes" if entry.get('success', True) else "No"
                    
                    self.transfer_tree.insert("", "end", values=(episode, source, target, types, success))
            except Exception as e:
                print(f"Error updating transfer plot: {e}")
                # Create a simple message if visualization fails
                ax = self.transfer_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Could not generate transfer visualization\nError: {str(e)}", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                self.transfer_canvas.draw()
        else:
            # Create a "no data" message
            ax = self.transfer_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No knowledge transfer data available", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.transfer_canvas.draw()
    
    def _update_metrics_plot(self):
        """Update the system metrics visualization."""
        # Clear previous figure
        self.metrics_fig.clear()
        
        # Create plot if we have data
        if hasattr(self, 'visualizer') and self.memory_usage:
            try:
                # Extract memory usage data
                episodes = []
                usage_values = []
                
                for bucket, usage in self.memory_usage.items():
                    for ep, val in usage.items():
                        episodes.append(int(ep))
                        usage_values.append(val)
                
                if episodes and usage_values:
                    # Sort by episode
                    sorted_pairs = sorted(zip(episodes, usage_values))
                    episodes = [pair[0] for pair in sorted_pairs]
                    usage_values = [pair[1] for pair in sorted_pairs]
                    
                    # Get transfer episodes
                    transfer_episodes = [entry.get('episode', 0) for entry in self.transfer_history]
                    
                    # Replace the figure with the one created by the visualizer
                    plt.close(self.metrics_fig)
                    self.metrics_fig = self.visualizer.plot_memory_usage(
                        episodes,
                        usage_values,
                        transfer_episodes=transfer_episodes,
                        title="Memory Usage During Training"
                    )
                    
                    # Update canvas
                    self.metrics_canvas.figure = self.metrics_fig
                    self.metrics_canvas.draw()
                else:
                    raise ValueError("Memory usage data is empty")
            except Exception as e:
                print(f"Error updating metrics plot: {e}")
                # Create a simple message if visualization fails
                ax = self.metrics_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Could not generate metrics visualization\nError: {str(e)}", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                self.metrics_canvas.draw()
        else:
            # Create a "no data" message
            ax = self.metrics_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No system metrics data available", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            self.metrics_canvas.draw()
    
    def _update_log(self, event=None):
        """Update the training log display."""
        bucket = self.log_bucket_var.get()
        if bucket == "All":
            buckets = self.active_buckets
        else:
            buckets = [bucket]
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Read and display logs for selected buckets
        for bucket in buckets:
            log_file = os.path.join(self.models_dir, bucket, "training_log.txt")
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        log_content = f.read()
                    
                    # Add bucket header
                    self.log_text.insert(tk.END, f"=== {bucket} Log ===\n", "bucket_header")
                    self.log_text.insert(tk.END, log_content)
                    self.log_text.insert(tk.END, "\n\n")
                except:
                    self.log_text.insert(tk.END, f"Error reading log for {bucket}\n\n")
        
        # Configure tags
        self.log_text.tag_configure("bucket_header", foreground="blue", font=("Helvetica", 10, "bold"))
    
    def start(self):
        """Start the monitoring application."""
        # Add protocol to handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        
        # Start the mainloop
        self.root.mainloop()
    
    def quit(self):
        """Clean up and quit the application."""
        # Stop the watching thread if active
        self.watching = False
        
        # Make sure all child windows are closed
        for child in self.root.winfo_children():
            if hasattr(child, 'destroy'):
                try:
                    child.destroy()
                except:
                    pass
        
        # Destroy the main window
        self.root.destroy()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Progressive Training Monitor")
    parser.add_argument("--models-dir", type=str, default=MODELS_DIR_DEFAULT,
                        help="Directory containing model folders")
    parser.add_argument("--update-interval", type=int, default=5,
                       help="Data refresh interval in seconds")
    
    args = parser.parse_args()
    
    # Start the monitoring UI
    monitor = TrainingMonitor(
        models_dir=args.models_dir,
        update_interval=args.update_interval
    )
    monitor.start()

if __name__ == "__main__":
    main() 