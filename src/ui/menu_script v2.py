import PySimpleGUI as sg
import json
import os
import subprocess
import psutil
import datetime
import threading
import queue

# File Paths and Constants
CONFIG_FILE = r"C:\Users\chris\OneDrive\Documents\GitHub\BTC-AI\AI Version 4\Scripts\final_config.json"
MODELS_DIR_BASE = r"C:\Users\chris\OneDrive\Documents\GitHub\BTC-AI\AI Version 4\Models"
NOTES_FILE = "notes.txt"
CHECKPOINT_INTERVAL = 10  # Save checkpoint every 10 episodes

# Default Configuration
default_config = {
    "LOOK_BACK_PERIOD": "1 week",
    "INITIAL_CAPITAL": 100000.0,
    "MAX_POSITIONS": 50,
    "BUCKET": "Scalping",
    "RESUME_CHECKPOINT": False,
    "MONTHLY_MIN": 15.0,
    "MONTHLY_MAX": 30.0,
    "YEARLY_MIN": 100.0,
    "YEARLY_MAX": 200.0,
}

# Load Config
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = default_config.copy()

# Load Notes
notes_content = ""
if os.path.exists(NOTES_FILE):
    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        notes_content = f.read()

# Bucket-Specific Presets
presets = {
    "Scalping": {"LOOK_BACK_PERIOD": "1 day", "INITIAL_CAPITAL": 50000.0, "MAX_POSITIONS": 20},
    "Short": {"LOOK_BACK_PERIOD": "1 week", "INITIAL_CAPITAL": 100000.0, "MAX_POSITIONS": 50},
    "Medium": {"LOOK_BACK_PERIOD": "2 weeks", "INITIAL_CAPITAL": 150000.0, "MAX_POSITIONS": 75},
    "Long": {"LOOK_BACK_PERIOD": "1 month", "INITIAL_CAPITAL": 200000.0, "MAX_POSITIONS": 100},
}

# GUI Layout Components
scalping_goals = [
    [sg.Text("Scalping Monthly Targets (%):")],
    [sg.Text("Min:"), sg.InputText(str(config.get("MONTHLY_MIN", 15.0)), key="MONTHLY_MIN", size=(5, 1)),
     sg.Text("Max:"), sg.InputText(str(config.get("MONTHLY_MAX", 30.0)), key="MONTHLY_MAX", size=(5, 1))],
]

other_goals = [
    [sg.Text("Yearly Targets (%):")],
    [sg.Text("Min:"), sg.InputText(str(config.get("YEARLY_MIN", 100.0)), key="YEARLY_MIN", size=(5, 1)),
     sg.Text("Max:"), sg.InputText(str(config.get("YEARLY_MAX", 200.0)), key="YEARLY_MAX", size=(5, 1))],
]

goals_frame = sg.Frame("Goals", [
    [sg.Column(scalping_goals, key="SCALPING_GOALS", visible=(config["BUCKET"] == "Scalping")),
     sg.Column(other_goals, key="OTHER_GOALS", visible=(config["BUCKET"] != "Scalping"))],
], relief=sg.RELIEF_SUNKEN)

settings_frame = sg.Frame("Settings", [
    [sg.Text("Trading Style:"),
     sg.Combo(["Scalping", "Short", "Medium", "Long"], default_value=config["BUCKET"], key="BUCKET", enable_events=True)],
    [sg.Text("Starting Cash ($):"), sg.InputText(str(config["INITIAL_CAPITAL"]), key="INITIAL_CAPITAL", size=(10, 1))],
    [sg.Text("Max Trades at Once:"), sg.InputText(str(config["MAX_POSITIONS"]), key="MAX_POSITIONS", size=(5, 1))],
    [sg.Text("Look-Back Period:"), sg.InputText(config["LOOK_BACK_PERIOD"], key="LOOK_BACK_PERIOD", size=(10, 1)),
     sg.Text("(e.g., '2 days', '1 month')", font=("Any", 8))],
    [sg.Button("Load Preset", key="LOAD_PRESET"), sg.Checkbox("Resume Last Run", default=config["RESUME_CHECKPOINT"], key="RESUME_CHECKPOINT")],
], relief=sg.RELIEF_SUNKEN)

run_frame = sg.Frame("Controls", [
    [sg.Button("Start Agent", key="RUN_AGENT"), sg.Button("Pause/Resume", key="PAUSE_RESUME", disabled=True),
     sg.Button("Stop Agent", key="STOP_AGENT", disabled=True)],
], relief=sg.RELIEF_SUNKEN)

log_frame = sg.Frame("Live Updates", [
    [sg.Multiline("", key="-LOG-", size=(60, 10), autoscroll=True, disabled=True)],
    [sg.Button("Pop-Out Log", key="OPEN_LOG_PIP"), sg.Button("Clear Log", key="CLEAR_LOG")],
])

notes_frame = sg.Frame("Notes", [
    [sg.Multiline(notes_content, key="-NOTES-", size=(60, 5), autoscroll=True, enable_events=True)],
    [sg.Button("Pop-Out Notes", key="OPEN_NOTES_PIP")],
])

layout = [
    [sg.Column([[settings_frame], [goals_frame], [run_frame]], vertical_scroll_only=True, expand_y=True),
     sg.VSeparator(),
     sg.Column([[log_frame], [notes_frame]], vertical_scroll_only=True, expand_y=True)],
]

# Initialize Main Window and PiP Windows
window = sg.Window("Trading Buddy", layout, resizable=True, finalize=True)
log_pip_window = None
notes_pip_window = None

# Helper Functions
def convert_time_to_bars(time_str):
    try:
        time_str = time_str.lower().strip()
        if "day" in time_str:
            days = float(time_str.split()[0])
            return int(days * 288)
        elif "week" in time_str:
            weeks = float(time_str.split()[0])
            return int(weeks * 7 * 288)
        elif "month" in time_str:
            months = float(time_str.split()[0])
            return int(months * 30 * 288)
        else:
            return int(time_str)
    except (ValueError, IndexError):
        sg.popup_error("Invalid Look-Back Period format. Use '2 days', '1 week', etc.")
        return None

def save_config_and_notes(config, notes):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        f.write(notes)

def read_process_output(process, log_queue):
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            log_queue.put(line.strip())

# Event Loop Variables
process = None
paused = False
log_queue = queue.Queue()
output_thread = None

# Event Loop
while True:
    window_, event, values = sg.read_all_windows(timeout=100)

    # Handle Window Closure
    if event in (sg.WIN_CLOSED, "Exit"):
        if window_ == window:
            if process and process.poll() is None:
                process.terminate()
            config.update({
                "BUCKET": values["BUCKET"],
                "INITIAL_CAPITAL": float(values["INITIAL_CAPITAL"] or 100000.0),
                "MAX_POSITIONS": int(values["MAX_POSITIONS"] or 50),
                "LOOK_BACK_PERIOD": values["LOOK_BACK_PERIOD"],
                "RESUME_CHECKPOINT": values["RESUME_CHECKPOINT"],
                "MONTHLY_MIN": float(values["MONTHLY_MIN"] or 15.0),
                "MONTHLY_MAX": float(values["MONTHLY_MAX"] or 30.0),
                "YEARLY_MIN": float(values["YEARLY_MIN"] or 100.0),
                "YEARLY_MAX": float(values["YEARLY_MAX"] or 200.0),
            })
            save_config_and_notes(config, values["-NOTES-"])
            break
        elif window_ == log_pip_window:
            log_pip_window.close()
            log_pip_window = None
        elif window_ == notes_pip_window:
            window["-NOTES-"].update(values["-NOTES_PIP-"])
            save_config_and_notes(config, values["-NOTES_PIP-"])
            notes_pip_window.close()
            notes_pip_window = None
        continue

    # Bucket Switch
    if event == "BUCKET":
        bucket = values["BUCKET"]
        window["SCALPING_GOALS"].update(visible=(bucket == "Scalping"))
        window["OTHER_GOALS"].update(visible=(bucket != "Scalping"))

    # Load Preset
    if event == "LOAD_PRESET":
        bucket = values["BUCKET"]
        if bucket in presets:
            preset = presets[bucket]
            window["LOOK_BACK_PERIOD"].update(preset["LOOK_BACK_PERIOD"])
            window["INITIAL_CAPITAL"].update(preset["INITIAL_CAPITAL"])
            window["MAX_POSITIONS"].update(preset["MAX_POSITIONS"])
            sg.popup(f"Loaded {bucket} preset!")

    # Start Agent
    if event == "RUN_AGENT":
        try:
            bars = convert_time_to_bars(values["LOOK_BACK_PERIOD"])
            if bars is None:
                continue
            config["LOOK_BACK_BARS"] = bars
            config.update({
                "BUCKET": values["BUCKET"],
                "INITIAL_CAPITAL": float(values["INITIAL_CAPITAL"]),
                "MAX_POSITIONS": int(values["MAX_POSITIONS"]),
                "RESUME_CHECKPOINT": values["RESUME_CHECKPOINT"],
                "MONTHLY_MIN": float(values["MONTHLY_MIN"]),
                "MONTHLY_MAX": float(values["MONTHLY_MAX"]),
                "YEARLY_MIN": float(values["YEARLY_MIN"]),
                "YEARLY_MAX": float(values["YEARLY_MAX"]),
            })
            save_config_and_notes(config, values["-NOTES-"])
            agent_script = r"C:\Users\chris\OneDrive\Documents\GitHub\BTC-AI\AI Version 4\Scripts\final_ai_agent.py"
            process = subprocess.Popen(["python", agent_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            window["RUN_AGENT"].update(disabled=True)
            window["PAUSE_RESUME"].update(disabled=False, text="Pause")
            window["STOP_AGENT"].update(disabled=False)
            log_queue = queue.Queue()
            output_thread = threading.Thread(target=read_process_output, args=(process, log_queue), daemon=True)
            output_thread.start()
        except Exception as e:
            sg.popup_error(f"Failed to start agent: {e}")

    # Pause/Resume
    if event == "PAUSE_RESUME" and process and process.poll() is None:
        try:
            proc = psutil.Process(process.pid)
            if not paused:
                proc.suspend()
                paused = True
                window["PAUSE_RESUME"].update(text="Resume")
                window["-LOG-"].print(f"[{datetime.datetime.now()}] Agent paused")
            else:
                proc.resume()
                paused = False
                window["PAUSE_RESUME"].update(text="Pause")
                window["-LOG-"].print(f"[{datetime.datetime.now()}] Agent resumed")
        except Exception as e:
            sg.popup_error(f"Pause/Resume failed: {e}")

    # Stop Agent
    if event == "STOP_AGENT" and process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
            window["-LOG-"].print(f"[{datetime.datetime.now()}] Agent stopped")
            process = None
            paused = False
            window["RUN_AGENT"].update(disabled=False)
            window["PAUSE_RESUME"].update(disabled=True, text="Pause")
            window["STOP_AGENT"].update(disabled=True)
        except Exception as e:
            sg.popup_error(f"Failed to stop agent: {e}")

    # Log PiP
    if event == "OPEN_LOG_PIP":
        if not log_pip_window:
            log_pip_layout = [
                [sg.Multiline("", key="-LOG_PIP-", size=(60, 10), autoscroll=True, disabled=True)],
                [sg.Button("Close", key="CLOSE_LOG_PIP")]
            ]
            log_pip_window = sg.Window("Live Log PiP", log_pip_layout, keep_on_top=True, finalize=True)
            log_pip_window["-LOG_PIP-"].update(window["-LOG-"].get())  # Sync initial content
        else:
            log_pip_window.bring_to_front()

    if window_ == log_pip_window and event == "CLOSE_LOG_PIP":
        log_pip_window.close()
        log_pip_window = None

    # Notes PiP
    if event == "OPEN_NOTES_PIP":
        if not notes_pip_window:
            notes_pip_layout = [
                [sg.Multiline(values["-NOTES-"], key="-NOTES_PIP-", size=(60, 5), autoscroll=True, enable_events=True)],
                [sg.Button("Close", key="CLOSE_NOTES_PIP")]
            ]
            notes_pip_window = sg.Window("Notes PiP", notes_pip_layout, keep_on_top=True, finalize=True)
        else:
            notes_pip_window.bring_to_front()

    if window_ == notes_pip_window and event == "-NOTES_PIP-":
        window["-NOTES-"].update(values["-NOTES_PIP-"])
        save_config_and_notes(config, values["-NOTES_PIP-"])

    if window_ == notes_pip_window and event == "CLOSE_NOTES_PIP":
        window["-NOTES-"].update(values["-NOTES_PIP-"])
        save_config_and_notes(config, values["-NOTES_PIP-"])
        notes_pip_window.close()
        notes_pip_window = None

    # Main Notes Sync
    if event == "-NOTES-":
        save_config_and_notes(config, values["-NOTES-"])
        if notes_pip_window:
            notes_pip_window["-NOTES_PIP-"].update(values["-NOTES-"])

    # Update Log
    try:
        while not log_queue.empty():
            line = log_queue.get_nowait()
            window["-LOG-"].print(line)
            if log_pip_window:
                log_pip_window["-LOG_PIP-"].print(line)
    except queue.Empty:
        pass

    if event == "CLEAR_LOG":
        window["-LOG-"].update("")
        if log_pip_window:
            log_pip_window["-LOG_PIP-"].update("")

# Cleanup
if log_pip_window:
    log_pip_window.close()
if notes_pip_window:
    notes_pip_window.close()
window.close()
