"""
═══════════════════════════════════════════════════════════════════════════
 COLAB AUTOPILOT — Server Cell
 
 Paste this into a NEW CELL at the TOP of your Colab notebook 
 (right after your dependency installs, before any model code).
 
 Then run it. It will:
   1. Install flask + cloudflared
   2. Start a background API server
   3. Open a Cloudflare tunnel  
   4. Print a connection string for your local MCP server
 
 Your notebook cells continue to work normally after this.
═══════════════════════════════════════════════════════════════════════════
"""

# ── Cell 1: Install & Setup ──────────────────────────────────────────
# !pip install -q flask cryptography
# !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared 2>/dev/null
# !chmod +x /usr/local/bin/cloudflared

# ── Cell 2: Autopilot Server ────────────────────────────────────────

import base64
import glob as glob_mod
import io
import json
import os
import secrets
import signal
import subprocess
import sys
import re
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone

from cryptography.fernet import Fernet, InvalidToken
from flask import Flask, request, jsonify

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG — Edit these to match your notebook paths
# ═══════════════════════════════════════════════════════════════════════
VERSION = "0.2.0"

# Where your training loop writes JSON logs (one JSON object per line)
# This should match TRAINING_LOGS_PATH in your training cell
TRAINING_LOG_PATH = "/content/drive/MyDrive/world_model/training_logs.jsonl"

# Where checkpoints are saved
CHECKPOINT_DIR = "/content/drive/MyDrive/world_model/weights"

# Working directory where your notebook code lives
WORK_DIR = "/content"

# ═══════════════════════════════════════════════════════════════════════
#  AUTH + ENCRYPTION
# ═══════════════════════════════════════════════════════════════════════
BEARER_TOKEN = secrets.token_hex(32)
FERNET_KEY = Fernet.generate_key()
START_TIME = time.time()
fernet = Fernet(FERNET_KEY)

# ═══════════════════════════════════════════════════════════════════════
#  BACKGROUND JOB MANAGER
#  Tracks long-running processes (like training) without blocking the API
# ═══════════════════════════════════════════════════════════════════════
_jobs = {}  # job_id -> {"process": Popen, "log_path": str, "started": float, ...}
_job_counter = 0
_jobs_lock = threading.Lock()

# Ring buffer for recent log lines per job (avoids reading huge files)
_job_recent_lines = {}  # job_id -> deque(maxlen=200)

def _tail_job_output(job_id, process, log_path):
    """Background thread that tails stdout/stderr to a log file and ring buffer."""
    buf = _job_recent_lines.get(job_id, deque(maxlen=500))
    _job_recent_lines[job_id] = buf
    
    with open(log_path, "w") as f:
        for stream in [process.stdout]:
            if stream is None:
                continue
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\n")
                timestamp = datetime.now().strftime("%H:%M:%S")
                entry = f"[{timestamp}] {line}"
                f.write(entry + "\n")
                f.flush()
                buf.append(entry)
    
    # Also capture stderr at the end
    if process.stderr:
        stderr_out = process.stderr.read()
        if stderr_out:
            with open(log_path, "a") as f:
                for line in stderr_out.strip().split("\n"):
                    entry = f"[STDERR] {line}"
                    f.write(entry + "\n")
                    buf.append(entry)


# ═══════════════════════════════════════════════════════════════════════
#  GPU INFO HELPER
# ═══════════════════════════════════════════════════════════════════════
def _gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": torch.cuda.get_device_name(0),
                "vram_total_gb": round(props.total_mem / 1024**3, 1),
                "vram_used_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
                "utilization": "use nvidia-smi for util%",
            }
        return {"name": "No GPU", "vram_total_gb": 0}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
#  FLASK API
# ═══════════════════════════════════════════════════════════════════════
app = Flask(__name__)

def decrypt_request():
    return json.loads(fernet.decrypt(request.data).decode("utf-8"))

def encrypt_response(data):
    return fernet.encrypt(json.dumps(data).encode("utf-8"))

@app.before_request
def auth_middleware():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {BEARER_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401
    return None


# ── Health / Status ──────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    disk = os.statvfs("/")
    disk_free = f"{disk.f_bavail * disk.f_frsize / 1024**3:.1f}GB"
    
    # Active jobs summary
    active_jobs = []
    with _jobs_lock:
        for jid, info in _jobs.items():
            proc = info["process"]
            active_jobs.append({
                "job_id": jid,
                "command": info["command"][:80],
                "running": proc.poll() is None,
                "pid": proc.pid,
                "started": info["started_str"],
            })
    
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "gpu": _gpu_info(),
        "disk_free": disk_free,
        "uptime_seconds": int(time.time() - START_TIME),
        "active_jobs": active_jobs,
        "python": sys.version.split()[0],
    })


# ── Execute shell command (short-running, waits for result) ──────────
@app.route("/exec", methods=["POST"])
def exec_cmd():
    data = decrypt_request()
    command = data["command"]
    timeout = min(data.get("timeout", 300), 600)
    t0 = time.time()
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=WORK_DIR,
        )
        return encrypt_response({
            "stdout": result.stdout[-10000:],  # Cap output to avoid token explosion
            "stderr": result.stderr[-5000:],
            "exit_code": result.returncode,
            "duration": round(time.time() - t0, 2),
            "truncated": len(result.stdout) > 10000,
        })
    except subprocess.TimeoutExpired:
        return encrypt_response({
            "stdout": "", "stderr": f"Timed out after {timeout}s",
            "exit_code": -1, "duration": timeout,
        })


# ── Execute Python code (short-running) ─────────────────────────────
@app.route("/python", methods=["POST"])
def exec_python():
    data = decrypt_request()
    code = data["code"]
    timeout = min(data.get("timeout", 300), 600)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=WORK_DIR,
        )
        return encrypt_response({
            "output": result.stdout[-10000:],
            "error": result.stderr[-5000:],
            "exit_code": result.returncode,
            "duration": round(time.time() - t0, 2),
        })
    except subprocess.TimeoutExpired:
        return encrypt_response({
            "output": "", "error": f"Timed out after {timeout}s",
            "exit_code": -1, "duration": timeout,
        })


# ── Submit background job (long-running, returns immediately) ────────
@app.route("/submit_job", methods=["POST"])
def submit_job():
    global _job_counter
    data = decrypt_request()
    command = data["command"]
    
    with _jobs_lock:
        _job_counter += 1
        job_id = f"job_{_job_counter}"
    
    log_path = f"/tmp/autopilot_{job_id}.log"
    
    process = subprocess.Popen(
        command, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=WORK_DIR,
        # Make it a process group so we can kill the whole tree
        preexec_fn=os.setsid,
    )
    
    info = {
        "process": process,
        "command": command,
        "log_path": log_path,
        "started": time.time(),
        "started_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with _jobs_lock:
        _jobs[job_id] = info
    
    # Start background tailer
    t = threading.Thread(target=_tail_job_output, args=(job_id, process, log_path), daemon=True)
    t.start()
    
    return encrypt_response({
        "job_id": job_id,
        "pid": process.pid,
        "log_path": log_path,
        "message": f"Job submitted. Use /job_status or /job_logs to monitor.",
    })


# ── Job status ───────────────────────────────────────────────────────
@app.route("/job_status", methods=["POST"])
def job_status():
    data = decrypt_request()
    job_id = data.get("job_id")
    
    with _jobs_lock:
        if job_id and job_id in _jobs:
            jobs_to_check = {job_id: _jobs[job_id]}
        else:
            jobs_to_check = dict(_jobs)
    
    results = {}
    for jid, info in jobs_to_check.items():
        proc = info["process"]
        running = proc.poll() is None
        results[jid] = {
            "running": running,
            "exit_code": proc.returncode if not running else None,
            "command": info["command"][:100],
            "started": info["started_str"],
            "elapsed_seconds": int(time.time() - info["started"]),
            "pid": proc.pid,
        }
    
    return encrypt_response(results)


# ── Job logs (last N lines from ring buffer — cheap!) ────────────────
@app.route("/job_logs", methods=["POST"])
def job_logs():
    data = decrypt_request()
    job_id = data["job_id"]
    n_lines = min(data.get("n_lines", 50), 200)  # Cap at 200 lines
    
    buf = _job_recent_lines.get(job_id, deque())
    lines = list(buf)[-n_lines:]
    
    return encrypt_response({
        "job_id": job_id,
        "n_lines": len(lines),
        "lines": lines,
        "total_buffered": len(buf),
    })


# ── Kill a job ───────────────────────────────────────────────────────
@app.route("/kill_job", methods=["POST"])
def kill_job():
    data = decrypt_request()
    job_id = data["job_id"]
    
    with _jobs_lock:
        info = _jobs.get(job_id)
    
    if not info:
        return encrypt_response({"error": f"No job {job_id}"})
    
    proc = info["process"]
    if proc.poll() is not None:
        return encrypt_response({"message": f"Job {job_id} already finished", "exit_code": proc.returncode})
    
    try:
        # Kill the entire process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(1)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return encrypt_response({"message": f"Killed job {job_id}", "pid": proc.pid})
    except Exception as e:
        return encrypt_response({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING-SPECIFIC ENDPOINTS
#  These read from YOUR training loop's JSON log file
# ═══════════════════════════════════════════════════════════════════════

@app.route("/training_summary", methods=["POST"])
def training_summary():
    """
    THE KEY ENDPOINT — Returns a compact metrics summary instead of raw logs.
    This is what prevents token explosion.
    
    Reads the JSONL log your training loop writes and returns:
    - Last N episodes with rewards
    - Rolling averages
    - Loss trends (sampled, not every point)
    - Anomaly flags (NaN, reward collapse, etc.)
    """
    data = decrypt_request()
    log_path = data.get("log_path", TRAINING_LOG_PATH)
    last_n = min(data.get("last_n_entries", 50), 200)
    
    if not os.path.exists(log_path):
        return encrypt_response({"error": f"Log file not found: {log_path}"})
    
    # Read last N lines efficiently (tail)
    try:
        entries = []
        with open(log_path, "r") as f:
            # Read all lines (they're small JSON objects)
            all_lines = f.readlines()
        
        for line in all_lines[-last_n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return encrypt_response({"error": f"Failed to read log: {e}"})
    
    if not entries:
        return encrypt_response({"status": "no_data", "message": "Training log is empty"})
    
    # ── Build compact summary ────────────────────────────────────────
    
    # Separate episode entries from training entries
    episode_entries = [e for e in entries if "reward" in e and "episode" in e]
    train_entries = [e for e in entries if "world_loss_z" in e]
    
    # Latest state
    latest = entries[-1]
    latest_step = latest.get("step", 0)
    
    # Episode stats
    rewards = [e["reward"] for e in episode_entries]
    avg20_values = [e.get("avg20", 0) for e in episode_entries if "avg20" in e]
    
    episode_summary = {}
    if rewards:
        import numpy as np
        episode_summary = {
            "total_episodes": episode_entries[-1].get("episode", 0) if episode_entries else 0,
            "last_5_rewards": [round(r, 3) for r in rewards[-5:]],
            "last_5_avg20": [round(a, 3) for a in avg20_values[-5:]],
            "reward_mean_last20": round(float(np.mean(rewards[-20:])), 3),
            "reward_std_last20": round(float(np.std(rewards[-20:])), 3),
            "reward_min_last20": round(float(np.min(rewards[-20:])), 3),
            "reward_max_last20": round(float(np.max(rewards[-20:])), 3),
            "crash_rate_last20": sum(1 for e in episode_entries[-20:] if e.get("end") == "CRASH") / min(20, len(episode_entries)),
        }
    
    # Loss trends (sample every Nth point to keep it compact)
    loss_summary = {}
    if train_entries:
        # Take at most 10 sampled points to show the trend
        step_size = max(1, len(train_entries) // 10)
        sampled = train_entries[::step_size][-10:]
        
        loss_summary = {
            "total_training_updates": len(train_entries),
            "trend": [
                {
                    "step": e.get("step"),
                    "wm_z": round(e.get("world_loss_z", 0), 6),
                    "wm_r": round(e.get("world_loss_r", 0), 4),
                    "actor": round(e.get("actor_loss", 0), 4),
                    "critic": round(e.get("critic_loss", 0), 4),
                }
                for e in sampled
            ],
        }
        
        # Compare first half vs second half for trend direction
        if len(train_entries) >= 4:
            mid = len(train_entries) // 2
            first_half = train_entries[:mid]
            second_half = train_entries[mid:]
            
            def avg_field(entries, field):
                vals = [e.get(field, 0) for e in entries if field in e]
                return float(np.mean(vals)) if vals else 0
            
            loss_summary["trend_direction"] = {
                "wm_z": "improving" if avg_field(second_half, "world_loss_z") < avg_field(first_half, "world_loss_z") else "degrading",
                "wm_r": "improving" if avg_field(second_half, "world_loss_r") < avg_field(first_half, "world_loss_r") else "degrading",
                "reward": "improving" if avg_field(second_half, "avg20") > avg_field(first_half, "avg20") else "degrading",
            }
    
    # ── Anomaly detection ────────────────────────────────────────────
    anomalies = []
    
    # Check for NaN in losses
    for e in entries[-10:]:
        for field in ["world_loss_z", "world_loss_r", "actor_loss", "critic_loss"]:
            val = e.get(field)
            if val is not None and (val != val or abs(val) > 1e6):  # NaN or explosion
                anomalies.append(f"CRITICAL: {field} = {val} at step {e.get('step')}")
    
    # Check for reward collapse
    if len(rewards) >= 10:
        recent_avg = float(np.mean(rewards[-10:]))
        if recent_avg < -50:
            anomalies.append(f"WARNING: Reward collapsed to avg={recent_avg:.1f} over last 10 episodes")
    
    # Check for 100% crash rate
    if episode_entries:
        recent_crashes = sum(1 for e in episode_entries[-10:] if e.get("end") == "CRASH")
        if recent_crashes == min(10, len(episode_entries)):
            anomalies.append("WARNING: 100% crash rate in last 10 episodes")
    
    # Check for stagnant world model
    if len(train_entries) >= 20:
        recent_wm = [e.get("world_loss_z", 0) for e in train_entries[-10:]]
        older_wm = [e.get("world_loss_z", 0) for e in train_entries[-20:-10]]
        if recent_wm and older_wm:
            if abs(float(np.mean(recent_wm)) - float(np.mean(older_wm))) < 1e-6:
                anomalies.append("WARNING: World model loss appears stagnant")
    
    summary = {
        "current_step": latest_step,
        "phase": latest.get("phase", latest.get("map", "unknown")),
        "total_log_entries": len(all_lines),
        "episodes": episode_summary,
        "losses": loss_summary,
        "anomalies": anomalies if anomalies else ["none"],
        "gpu": _gpu_info(),
    }
    
    return encrypt_response(summary)


@app.route("/training_logs_raw", methods=["POST"])
def training_logs_raw():
    """Return last N raw JSONL entries. Use sparingly — prefer /training_summary."""
    data = decrypt_request()
    log_path = data.get("log_path", TRAINING_LOG_PATH)
    last_n = min(data.get("last_n", 20), 50)
    
    if not os.path.exists(log_path):
        return encrypt_response({"error": f"Log not found: {log_path}"})
    
    with open(log_path, "r") as f:
        lines = f.readlines()
    
    entries = []
    for line in lines[-last_n:]:
        try:
            entries.append(json.loads(line.strip()))
        except:
            continue
    
    return encrypt_response({"entries": entries, "total_lines": len(lines)})


@app.route("/list_checkpoints", methods=["POST"])
def list_checkpoints():
    """List saved model checkpoints."""
    data = decrypt_request()
    ckpt_dir = data.get("checkpoint_dir", CHECKPOINT_DIR)
    
    if not os.path.exists(ckpt_dir):
        return encrypt_response({"error": f"Dir not found: {ckpt_dir}", "checkpoints": []})
    
    files = []
    for f in sorted(os.listdir(ckpt_dir)):
        fpath = os.path.join(ckpt_dir, f)
        if os.path.isfile(fpath):
            files.append({
                "name": f,
                "size_mb": round(os.path.getsize(fpath) / 1024**2, 1),
                "modified": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M"),
            })
    
    return encrypt_response({"checkpoint_dir": ckpt_dir, "files": files})


# ── File read/write (for the agent to edit notebook code) ────────────
@app.route("/read_file", methods=["POST"])
def read_file():
    data = decrypt_request()
    path = data["path"]
    max_bytes = min(data.get("max_bytes", 50000), 100000)
    
    if not os.path.exists(path):
        return encrypt_response({"error": f"Not found: {path}"})
    
    size = os.path.getsize(path)
    with open(path, "r", errors="replace") as f:
        content = f.read(max_bytes)
    
    return encrypt_response({
        "path": path,
        "content": content,
        "size_bytes": size,
        "truncated": size > max_bytes,
    })


@app.route("/write_file", methods=["POST"])
def write_file():
    data = decrypt_request()
    path = data["path"]
    content = data["content"]
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    
    return encrypt_response({
        "path": path,
        "size_bytes": len(content.encode("utf-8")),
        "message": "File written successfully",
    })


@app.route("/upload", methods=["POST"])
def upload():
    data = decrypt_request()
    path = data["path"]
    content_b64 = data["content_b64"]
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    content = base64.b64decode(content_b64)
    with open(path, "wb") as f:
        f.write(content)
    
    return encrypt_response({
        "path": path,
        "size_bytes": len(content),
    })


@app.route("/download", methods=["POST"])
def download():
    data = decrypt_request()
    path = data["path"]
    
    if not os.path.exists(path):
        return encrypt_response({"error": f"Not found: {path}"})
    
    size = os.path.getsize(path)
    if size > 50 * 1024 * 1024:
        return encrypt_response({"error": f"File too large: {size/1024**2:.1f}MB (max 50MB)"})
    
    with open(path, "rb") as f:
        content = f.read()
    
    return encrypt_response({
        "path": path,
        "content_b64": base64.b64encode(content).decode(),
        "size_bytes": size,
    })


# ═══════════════════════════════════════════════════════════════════════
#  START SERVER
# ═══════════════════════════════════════════════════════════════════════
def _run_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True)

flask_thread = threading.Thread(target=_run_flask, daemon=True)
flask_thread.start()
print(f"Autopilot API running on port 5000")

# Don't print secrets here — Cell 3 handles that after tunnel is up
