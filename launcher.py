import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
import os
import time
import glob
import threading
import json

class CarlaLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("CARLA Lane Departure Warning Launcher")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Config file path
        self.config_file = "launcher_config.json"

        # Load saved config
        saved_config = self.load_config()

        # Variables
        self.carla_path = tk.StringVar(value=saved_config.get("carla_path", "./CarlaUE4.exe"))
        self.quality_level = tk.StringVar(value="Low")
        self.run_mode = tk.StringVar(value="Normal")
        self.test_file = tk.StringVar()
        self.commands_dir = tk.StringVar(value="./test_commands")
        self.carla_process = None
        self.python_script = "manual_control_steeringwheel.py"

        # Create UI
        self.create_ui()

    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
        return {}

    def save_config(self):
        """Save configuration to file"""
        config = {
            "carla_path": self.carla_path.get()
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Error saving config: {e}")

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # CARLA settings section
        carla_frame = ttk.LabelFrame(main_frame, text="CARLA Settings")
        carla_frame.pack(fill=tk.X, padx=5, pady=5)

        # CARLA path selection
        ttk.Label(carla_frame, text="CARLA Executable:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(carla_frame, textvariable=self.carla_path, width=50).grid(column=1, row=0, padx=5, pady=5)
        ttk.Button(carla_frame, text="Browse", command=self.browse_carla).grid(column=2, row=0, padx=5, pady=5)

        # Quality level selection
        ttk.Label(carla_frame, text="Quality Level:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        quality_combo = ttk.Combobox(carla_frame, textvariable=self.quality_level, state="readonly")
        quality_combo['values'] = ('Low', 'Epic')
        quality_combo.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)

        # Script settings section
        script_frame = ttk.LabelFrame(main_frame, text="Script Settings")
        script_frame.pack(fill=tk.X, padx=5, pady=5)

        # Mode selection
        ttk.Label(script_frame, text="Run Mode:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        normal_radio = ttk.Radiobutton(script_frame, text="Normal Mode", variable=self.run_mode, value="Normal",
                                      command=self.toggle_test_options)
        normal_radio.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)

        test_radio = ttk.Radiobutton(script_frame, text="Test Mode", variable=self.run_mode, value="Test",
                                    command=self.toggle_test_options)
        test_radio.grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)

        # Test file options frame (initially disabled)
        self.test_options_frame = ttk.Frame(script_frame)
        self.test_options_frame.grid(column=0, row=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Label(self.test_options_frame, text="Commands Directory:").pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Entry(self.test_options_frame, textvariable=self.commands_dir, width=30).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(self.test_options_frame, text="Browse", command=self.browse_commands_dir).pack(side=tk.LEFT, padx=5, pady=5)

        # Available test files
        self.test_files_frame = ttk.LabelFrame(script_frame, text="Available Test Files")
        self.test_files_frame.grid(column=0, row=2, columnspan=3, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)

        # Test files listbox with scrollbar
        self.test_files_listbox = tk.Listbox(self.test_files_frame, height=10, width=70)
        self.test_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(self.test_files_frame, orient=tk.VERTICAL, command=self.test_files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_files_listbox.config(yscrollcommand=scrollbar.set)
        self.test_files_listbox.bind('<<ListboxSelect>>', self.on_test_file_select)

        # Set initial state
        self.toggle_test_options()

        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=10)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(button_frame, text="Launch", command=self.launch_application).pack(side=tk.RIGHT, padx=5)

        # Configure style
        style = ttk.Style()
        if 'Accent.TButton' not in style.theme_names():
            pass  # Apply custom style if needed

    def browse_carla(self):
        """Open file browser dialog specifically for finding CarlaUE4.exe"""
        filename = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.carla_path.get()) or ".",
            title="Select CarlaUE4.exe",
            filetypes=(("CARLA executable", "CarlaUE4.exe"), ("All files", "*.*"))
        )
        if filename:
            self.carla_path.set(filename)
            self.save_config()  # Save the path when changed

    def browse_commands_dir(self):
        directory = filedialog.askdirectory(
            initialdir=self.commands_dir.get() or ".",
            title="Select Commands Directory"
        )
        if directory:
            self.commands_dir.set(directory)
            self.refresh_test_files()

    def refresh_test_files(self):
        """Scan the commands directory for JSON files and update the listbox"""
        self.test_files_listbox.delete(0, tk.END)

        directory = self.commands_dir.get()
        if os.path.isdir(directory):
            json_files = glob.glob(os.path.join(directory, "*.json"))
            for file in json_files:
                self.test_files_listbox.insert(tk.END, os.path.basename(file))

    def on_test_file_select(self, event):
        """Handle test file selection from the listbox"""
        if self.test_files_listbox.curselection():
            selected_filename = self.test_files_listbox.get(self.test_files_listbox.curselection()[0])
            full_path = os.path.join(self.commands_dir.get(), selected_filename)
            self.test_file.set(full_path)

    def toggle_test_options(self):
        """Show or hide test options based on run mode"""
        if self.run_mode.get() == "Test":
            self.test_options_frame.grid()
            self.test_files_frame.grid()
            self.refresh_test_files()
        else:
            self.test_options_frame.grid_remove()
            self.test_files_frame.grid_remove()
            self.test_file.set("")

    def launch_application(self):
        """Launch CARLA and the script with appropriate parameters"""
        # Validate inputs
        carla_exe = self.carla_path.get()
        if not os.path.isfile(carla_exe):
            self.status_var.set(f"Error: {carla_exe} not found")
            return

        # Launch in a separate thread to keep UI responsive
        threading.Thread(target=self._launch_process, daemon=True).start()

    def _launch_process(self):
        """Worker thread for launching processes"""
        try:
            # Update status
            self.root.after(0, lambda: self.status_var.set("Starting CARLA..."))

            # Prepare CARLA command
            quality_arg = f"-quality-level={self.quality_level.get()}"
            carla_cmd = [self.carla_path.get(), quality_arg]

            # Start CARLA
            self.carla_process = subprocess.Popen(carla_cmd)

            # Wait for CARLA to initialize
            self.root.after(0, lambda: self.status_var.set("Waiting for CARLA to initialize..."))
            time.sleep(10)  # Give CARLA time to start

            # Prepare script command
            script_cmd = ["python", self.python_script]

            # Add test mode parameters if selected
            if self.run_mode.get() == "Test" and self.test_file.get():
                script_cmd.extend(["--test", "--test-file", self.test_file.get()])

            # Update status and run script
            self.root.after(0, lambda: self.status_var.set("Starting script..."))
            script_process = subprocess.Popen(script_cmd)
            script_process.wait()

            # Cleanup after script exit
            self.root.after(0, lambda: self.status_var.set("Script finished. Closing CARLA..."))
            if self.carla_process:
                self.carla_process.terminate()
                self.carla_process = None

            self.root.after(0, lambda: self.status_var.set("Ready"))

        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            if self.carla_process:
                self.carla_process.terminate()
                self.carla_process = None

    def on_closing(self):
        """Clean up when the application is closing"""
        self.save_config()  # Save config on exit
        if self.carla_process:
            try:
                self.carla_process.terminate()
                self.carla_process = None
            except:
                pass
        self.root.destroy()

if __name__ == "__main__":
    # Create test commands directory if it doesn't exist
    os.makedirs("test_commands", exist_ok=True)

    # Create sample command file if directory is empty
    if not os.listdir("test_commands"):
        sample_commands = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.9, 0.0, 0.4],
            [0.9, 0.0, 0.6],
            [0.7, 0.0, 0.6],
            [0.5, 0.0, 0.6],
            [0.3, 0.0, 0.6],
            [0.3, 0.0, 0.4],
            [0.3, 0.0, 0.2],
            [0.3, 0.0, 0.0],
            [0.3, 0.0, -0.2],
            [0.3, 0.0, -0.4],
            [0.3, 0.0, -0.6],
            [0.5, 0.0, -0.6],
            [0.7, 0.0, -0.6],
            [0.9, 0.0, -0.6],
            [0.9, 0.0, -0.4],
            [0.9, 0.0, -0.2],
            [0.9, 0.0, 0.0],
            [0.0, 0.5, 0.0]
        ]
        with open("test_commands/sample.json", "w") as f:
            json.dump(sample_commands, f)

    root = tk.Tk()
    app = CarlaLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()