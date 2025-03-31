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
        self.root.geometry("600x650")
        self.root.resizable(True, True)

        # Config file path
        self.config_file = "launcher_config.json"

        # Load saved config
        saved_config = self.load_config()

        # Variables
        self.carla_path = tk.StringVar(value=saved_config.get("carla_path", "./CarlaUE4.exe"))
        self.quality_level = tk.StringVar(value="Low")
        self.run_mode = tk.StringVar(value="Normal")
        self.controller = tk.StringVar(value="xbox")
        self.test_file = tk.StringVar()
        self.commands_dir = tk.StringVar(value="./test_commands")
        self.carla_process = None
        
        # Nuove variabili per la selezione del programma e modalità
        self.program_type = tk.StringVar(value="async")
        self.record_mode = tk.BooleanVar(value=False)

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



    def toggle_test_options(self):
        """Show or hide test options based on run mode"""
        if self.program_type.get() in ["sync", "sync_record"] and self.run_mode.get() == "Test":
            self.test_options_frame.grid()
            self.test_files_frame.grid()
            self.refresh_test_files()
        else:
            self.test_options_frame.grid_remove()
            self.test_files_frame.grid_remove()

    def toggle_controller_options(self):
        """Show or hide controller options based on program type and run mode"""
        # Show controller options for async modes that support controllers
        if self.program_type.get() in ["async", "async_controller"]:
            self.controller_selection_frame.grid()

            # Set default controller if not already set
            if not self.controller.get():
                self.controller.set("wheel")

        else:
            # Hide controller options for modes that don't support controllers
            self.controller_selection_frame.grid_remove()
            self.test_options_frame.grid_remove()
            self.test_files_frame.grid_remove()

    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # CARLA settings section
        carla_frame = ttk.LabelFrame(main_frame, text="CARLA Settings")
        carla_frame.pack(fill=tk.X, padx=5, pady=5)

        # CARLA path row
        carla_path_frame = ttk.Frame(carla_frame)
        carla_path_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(carla_path_frame, text="CARLA Path:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(carla_path_frame, textvariable=self.carla_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(carla_path_frame, text="Browse", command=self.browse_carla).pack(side=tk.LEFT, padx=5)

        # Quality level row
        quality_frame = ttk.Frame(carla_frame)
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(quality_frame, text="Quality Level:").pack(side=tk.LEFT, padx=5)
        for quality in ["Low", "Epic"]:
            ttk.Radiobutton(quality_frame, text=quality, variable=self.quality_level, value=quality).pack(side=tk.LEFT, padx=5)

        # Script settings section
        script_frame = ttk.LabelFrame(main_frame, text="Script Settings")
        script_frame.pack(fill=tk.X, padx=5, pady=5)

        # Use grid for script_frame children
        script_frame.columnconfigure(0, weight=1)

        # Program selection frame
        program_frame = ttk.LabelFrame(script_frame, text="Program Selection")
        program_frame.grid(row=0, column=0, sticky=tk.W+tk.E, padx=5, pady=5)

        # Radio buttons in program frame
        ttk.Radiobutton(program_frame, text="Async Mode", variable=self.program_type,
                       value="async", command=self.toggle_run_mode).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(program_frame, text="Sync Mode", variable=self.program_type,
                       value="sync", command=self.toggle_run_mode).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Controller selection - shown only for async mode
        self.controller_selection_frame = ttk.LabelFrame(script_frame, text="Controller Selection")
        self.controller_selection_frame.grid(row=1, column=0, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Radiobutton(self.controller_selection_frame, text="Xbox One", variable=self.controller,
                       value="xbox", command=self.toggle_controller_options).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(self.controller_selection_frame, text="G29", variable=self.controller,
                       value="wheel", command=self.toggle_controller_options).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(self.controller_selection_frame, text="Keyboard", variable=self.controller,
                       value="keyboard", command=self.toggle_controller_options).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

        # Run mode section - shown only for sync mode
        self.run_mode_frame = ttk.LabelFrame(script_frame, text="Run Mode")
        self.run_mode_frame.grid(row=2, column=0, sticky=tk.W+tk.E, padx=5, pady=5)

        ttk.Radiobutton(self.run_mode_frame, text="Record Mode", variable=self.run_mode,
                       value="sync_record", command=self.toggle_test_options).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(self.run_mode_frame, text="Test Mode", variable=self.run_mode,
                       value="sync_test", command=self.toggle_test_options).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Test file options frame (initially hidden)
        self.test_options_frame = ttk.LabelFrame(script_frame, text="Test Options")
        self.test_options_frame.grid(row=3, column=0, sticky=tk.W+tk.E, padx=5, pady=5)

        commands_frame = ttk.Frame(self.test_options_frame)
        commands_frame.pack(fill=tk.X, expand=True)
        ttk.Label(commands_frame, text="Commands Directory:").pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Entry(commands_frame, textvariable=self.commands_dir, width=30).pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        ttk.Button(commands_frame, text="Browse", command=self.browse_commands_dir).pack(side=tk.LEFT, padx=5, pady=5)

        # Available test files
        self.test_files_frame = ttk.LabelFrame(script_frame, text="Available Test Files")
        self.test_files_frame.grid(row=4, column=0, sticky=tk.W+tk.E+tk.N+tk.S, padx=5, pady=5)

        # Test files listbox with scrollbar
        self.test_files_listbox = tk.Listbox(self.test_files_frame, height=10)
        self.test_files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar = ttk.Scrollbar(self.test_files_frame, orient=tk.VERTICAL, command=self.test_files_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.test_files_listbox.config(yscrollcommand=scrollbar.set)
        self.test_files_listbox.bind('<<ListboxSelect>>', self.on_test_file_select)

        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=10)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(button_frame, text="Launch", command=self.launch_application).pack(side=tk.RIGHT, padx=5)

        # Initialize UI state
        self.toggle_run_mode()
        self.toggle_test_options()

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

    def toggle_run_mode(self):
        """Show or hide run mode options based on program type"""
        if self.program_type.get() == "async":
            self.controller_selection_frame.grid()
            self.run_mode_frame.grid_remove()
            self.test_options_frame.grid_remove()
            self.test_files_frame.grid_remove()
        elif self.program_type.get() == "sync":
            self.controller_selection_frame.grid_remove()
            self.run_mode_frame.grid()
            self.toggle_test_options()  # Update test options visibility
        else:
            self.controller_selection_frame.grid_remove()
            self.run_mode_frame.grid_remove()
            self.test_options_frame.grid_remove()
            self.test_files_frame.grid_remove()

    def toggle_test_options(self):
        """Show or hide test options based on run mode"""
        if self.program_type.get() == "sync" and self.run_mode.get() == "sync_test":
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

            # Determine which script to run based on program_type
            if self.program_type.get() == "async" and self.controller.get() == "xbox":
                script_name = "camera_lanes_analysis_async.py"
                script_args = ["--controller", "xbox"]
            elif self.program_type.get() == "async" and self.controller.get() == "wheel":
                script_name = "camera_lanes_analysis_async.py"
                script_args = ["--controller", "wheel"]
            elif self.program_type.get() == "async" and self.controller.get() == "keyboard":
                script_name = "camera_lanes_analysis_async.py"
                script_args = ["--controller", "keyboard"]
            elif self.program_type.get() == "sync" and self.run_mode.get() == "sync_record":
                script_name = "camera_lanes_analysis_sync.py"
                script_args = ["--record"]
            elif self.program_type.get() == "sync" and self.run_mode.get() == "sync_test":
                script_name = "camera_lanes_analysis_sync.py"
                # Pass the selected file as playback
                selected_file = os.path.basename(self.test_file.get())  # e.g. test_multi_lane_synch_30fps.json
                script_args = ["--playback", selected_file]
            else:
                script_name = "camera_lanes_analysis_sync.py"
                script_args = []

            # Prepare script command
            script_cmd = ["python", script_name] + script_args



            # Update status and run script
            self.root.after(0, lambda: self.status_var.set(f"Starting script {script_name}..."))
            print(f"Running command: {' '.join(script_cmd)}")
            script_process = subprocess.Popen(script_cmd)
            script_process.wait()

            # Cleanup after script exit
            self.root.after(0, lambda: self.status_var.set("Script finished. Closing CARLA..."))
            if self.carla_process:
                self.carla_process.terminate()
                self.carla_process = None

            self.root.after(0, lambda: self.status_var.set("Ready"))

        except Exception as e:
            error_msg = str(e)  # Capture the error message
            self.root.after(0, lambda msg=error_msg: self.status_var.set(f"Error: {msg}"))
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

    root = tk.Tk()
    app = CarlaLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()