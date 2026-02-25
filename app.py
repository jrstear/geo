#!/usr/bin/env python3
import os
import subprocess
import threading
import queue
import time
import sys
import re
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__)

# Path to the package.py script (assumed to be in the same directory)
PACKAGE_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "package.py"))

# Persistent storage for process logs
logs_queue = queue.Queue()

def get_defaults():
    """Parses package.py to extract default values for CLI arguments."""
    defaults = {}
    if not os.path.exists(PACKAGE_SCRIPT):
        return defaults

    try:
        with open(PACKAGE_SCRIPT, 'r') as f:
            content = f.read()
            # Match patterns like: add_argument("--scale", type=float, default=1.0, ...)
            # We look for --arg-name and default=value
            matches = re.finditer(r'add_argument\("--([^"]+)",[^)]*default=([^,)]+)', content)
            for m in matches:
                name = m.group(1).replace('-', '_')
                val = m.group(2).strip()
                # Clean up quotes if present
                val = val.strip("'\"")
                defaults[name] = val
    except Exception as e:
        print(f"⚠️ Warning: Could not parse defaults from package.py: {e}")

    return defaults

def run_package_command(args):
    """Runs the package.py command and puts output into the logs queue."""
    if not os.path.exists(PACKAGE_SCRIPT):
        msg = f"❌ Error: Script not found at {PACKAGE_SCRIPT}\n"
        print(msg)
        logs_queue.put(msg)
        logs_queue.put("EOF\n")
        return

    cmd = [sys.executable, PACKAGE_SCRIPT] + args
    print(f"🚀 Launching command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        for line in process.stdout:
            print(f"  [OUT] {line.strip()}")
            logs_queue.put(line)

        process.wait()
        print(f"✅ Process finished with code {process.returncode}")

        if process.returncode == 0:
            logs_queue.put("✨ Success! Task completed.\n")
        else:
            logs_queue.put(f"❌ Process exited with error code {process.returncode}\n")

    except Exception as e:
        msg = f"❌ Critical Error: {str(e)}\n"
        print(msg)
        logs_queue.put(msg)
    finally:
        logs_queue.put("EOF\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/defaults")
def defaults():
    return jsonify(get_defaults())

@app.route("/run", methods=["POST"])
def run():
    data = request.json
    print(f"📥 Received run request with data: {data}")

    # Validate: at least one input required
    tif_file     = (data.get("tif_file")     or "").strip()
    contour_file = (data.get("contour_file") or "").strip()
    tin_file     = (data.get("tin_file")     or "").strip()

    if not tif_file and not contour_file and not tin_file:
        return jsonify({"error": "At least one input file (TIF, DXF, or XML) is required."}), 400

    pkg_defaults = get_defaults()
    args = []

    def add_arg(key, flag):
        val = data.get(key)
        if val is not None and str(val).strip() != "":
            default_val = pkg_defaults.get(key)
            if default_val is None or str(val).strip() != str(default_val).strip():
                args.extend([flag, str(val)])

    # ── TIF positional arg + TIF-specific options ──────────────────────────────
    if tif_file:
        args.append(tif_file)

        if data.get("output_dir"):
            args += ["--output-dir", data["output_dir"]]

        if data.get("clobber"):
            args.append("--clobber")

        add_arg("downsize_percent", "--downsize-percent")
        add_arg("tile_size", "--tile-size")
        add_arg("load", "--load")

    # ── Shared transform args (apply to all input types) ──────────────────────
    add_arg("scale",    "--scale")
    add_arg("anchor_x", "--anchor-x")
    add_arg("anchor_y", "--anchor-y")
    add_arg("shift_x",  "--shift-x")
    add_arg("shift_y",  "--shift-y")

    # ── Contour DXF args ───────────────────────────────────────────────────────
    if contour_file:
        args += ["--contour-file", contour_file]
        add_arg("contour_suffix", "--contour-suffix")

    # ── TIN XML args ───────────────────────────────────────────────────────────
    if tin_file:
        args += ["--tin-file", tin_file]
        add_arg("tin_suffix",     "--tin-suffix")
        add_arg("tin_max_mb",     "--tin-max-mb")
        if (data.get("tin_output_dir") or "").strip():
            args += ["--tin-output-dir", data["tin_output_dir"].strip()]

    # Final command for display
    full_command = f"{os.path.basename(sys.executable)} {os.path.basename(PACKAGE_SCRIPT)} {' '.join(args)}"

    # Start the process in a background thread
    print("🧹 Clearing logs queue...")
    while not logs_queue.empty():
        logs_queue.get()

    print("🧵 Starting background thread...")
    thread = threading.Thread(target=run_package_command, args=(args,))
    thread.start()

    return jsonify({"status": "started", "command": full_command})


@app.route("/stream")
def stream():
    def event_stream():
        while True:
            line = logs_queue.get()
            if line.strip() == "EOF":
                yield f"data: {line}\n\n"
                break

            # SSE protocol: each line of a multi-line message must start with 'data: '
            # We split by '\n' and send a data: line for every single part.
            # For example, if line is "\n🚀 Tiling\n", parts will be ["", "🚀 Tiling", ""]
            # Sending these as data segments ensures the browser gets "\n🚀 Tiling\n" total.
            parts = line.split('\n')
            for part in parts:
                yield f"data: {part}\n"

            # Send the double-newline to terminate the SSE message
            yield "\n"

    return Response(event_stream(), mimetype="text/event-stream")

@app.route("/browse")
def browse():
    """Triggers a native file/folder picker using tkinter in a separate process to avoid macOS threading crashes."""
    mode = request.args.get("mode", "dir")

    # Build the filedialog call based on mode
    if mode == "tif":
        dialog_call = 'filedialog.askopenfilename(title="Select Input GeoTIFF", filetypes=[("GeoTIFF", "*.tif *.tiff")])'
    elif mode == "dxf":
        dialog_call = 'filedialog.askopenfilename(title="Select Contour DXF", filetypes=[("DXF", "*.dxf")])'
    elif mode == "xml":
        dialog_call = 'filedialog.askopenfilename(title="Select TIN LandXML", filetypes=[("LandXML", "*.xml")])'
    else:  # dir
        dialog_call = 'filedialog.askdirectory(title="Select Output Directory")'

    # Python script to run in a subprocess
    # This ensures tkinter runs on its own main thread
    script = f"""
import tkinter as tk
from tkinter import filedialog
import sys
import os
import platform

root = tk.Tk()
root.withdraw()

# Platform-specific window focus
system_platform = platform.system()
if system_platform == "Darwin":
    # Force this process to the front on macOS
    os.system(f'''/usr/bin/osascript -e 'tell application "System Events" to set frontmost of every process whose unix id is {{os.getpid()}} to true' ''')
elif system_platform == "Windows":
    # Lift and force focus on Windows
    root.lift()
    root.focus_force()

root.attributes('-topmost', True)

path = {dialog_call}

root.destroy()
if path:
    print(path)
"""
    try:
        # Run the script and capture the selected path
        output = subprocess.check_output(["python3", "-c", script], text=True).strip()
        return jsonify({"path": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
