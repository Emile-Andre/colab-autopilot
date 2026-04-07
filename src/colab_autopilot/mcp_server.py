"""MCP server for colab-autopilot — exposes Colab GPU as Claude Code tools.

Run via: colab-autopilot mcp-serve

Tools designed for AUTONOMOUS RL training workflows:
  - colab_status          → GPU/VRAM/jobs overview
  - colab_training_summary → Compact metrics digest (NOT raw logs)  
  - colab_exec            → Run shell commands
  - colab_python          → Run Python snippets
  - colab_submit_job      → Launch long-running training (returns immediately)
  - colab_job_logs        → Tail last N lines of a running job
  - colab_kill_job        → Stop a running job
  - colab_read_file       → Read a file on Colab
  - colab_write_file      → Write/edit a file on Colab  
  - colab_upload          → Upload local file to Colab
  - colab_download        → Download file from Colab
  - colab_list_checkpoints → List saved model weights
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, Union

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from colab_autopilot import __version__
from colab_autopilot.client import ColabClient, ColabError
from colab_autopilot.config import load_config

HEALTH_CHECK_INTERVAL = 60
_last_health_check: float = 0


def _get_client() -> Tuple[Optional[ColabClient], Optional[dict]]:
    config = load_config()
    if config is None:
        return None, None
    return ColabClient(
        url=config["url"],
        token=config["token"],
        encryption_key=config["encryption_key"],
    ), config


def _prepare_call() -> Tuple[Optional[ColabClient], list]:
    client, config = _get_client()
    if client is None:
        return None, ["Not connected. Run: colab-autopilot connect cc://TOKEN:KEY@host"]
    
    warnings = []
    # Check session age
    connected_at = config.get("connected_at", "")
    try:
        connected_dt = datetime.fromisoformat(connected_at.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - connected_dt).total_seconds() / 3600
        if age_hours > 10:
            warnings.append(f"WARNING: Session is {age_hours:.0f}h old (Colab max is 12h). Consider restarting.")
    except (ValueError, AttributeError):
        pass
    
    # Lazy health check
    global _last_health_check
    if time.time() - _last_health_check > HEALTH_CHECK_INTERVAL:
        try:
            client.health()
            _last_health_check = time.time()
        except ColabError as e:
            warnings.append(f"Health check failed: {e}")
    
    return client, warnings


def _call(fn, *args, **kwargs) -> str:
    """Wrapper that handles errors and formats response."""
    client, warnings = _prepare_call()
    if client is None:
        return warnings[0]
    try:
        result = fn(client, *args, **kwargs)
        if isinstance(result, dict):
            if warnings:
                result["_warnings"] = warnings
            return json.dumps(result, indent=2)
        return str(result)
    except ColabError as e:
        return f"Error: {e}"


def create_server() -> Server:
    server = Server("colab-autopilot")

    @server.list_tools()
    async def list_tools():
        return [
            # ── Status & Monitoring ──────────────────────────────────
            Tool(
                name="colab_status",
                description=(
                    "Get Colab GPU status: GPU name, VRAM usage, disk space, uptime, "
                    "and list of active background jobs. Use this first to check if "
                    "the Colab runtime is alive."
                ),
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            Tool(
                name="colab_training_summary",
                description=(
                    "Get a COMPACT training metrics summary — reward trends, loss trends, "
                    "anomaly detection (NaN, reward collapse, 100% crash rate, stagnant loss). "
                    "This reads the structured JSONL log your training loop writes. "
                    "ALWAYS prefer this over raw logs to save tokens. "
                    "Returns: current step, phase, last 5 rewards, rolling avg, "
                    "sampled loss curve (10 points), trend direction, and anomaly flags."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "last_n_entries": {
                            "type": "integer",
                            "description": "How many recent log entries to analyze (default 50, max 200)",
                            "default": 50,
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="colab_training_logs_raw",
                description=(
                    "Get raw JSONL training log entries. Use sparingly — prefer "
                    "colab_training_summary for routine monitoring. Only use this when "
                    "you need exact values from specific entries for debugging."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "last_n": {
                            "type": "integer",
                            "description": "Number of recent entries (default 20, max 50)",
                            "default": 20,
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="colab_list_checkpoints",
                description="List saved model checkpoint files with sizes and dates.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "checkpoint_dir": {
                            "type": "string",
                            "description": "Override checkpoint directory path (optional)",
                        },
                    },
                    "required": [],
                },
            ),

            # ── Job Management ───────────────────────────────────────
            Tool(
                name="colab_submit_job",
                description=(
                    "Submit a LONG-RUNNING command (like training) that runs in the background. "
                    "Returns immediately with a job_id. Use colab_job_logs to monitor, "
                    "colab_kill_job to stop. The command runs in a subprocess — "
                    "for training, use: python /path/to/train.py"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to run in background (e.g., 'python train.py')",
                        },
                    },
                    "required": ["command"],
                },
            ),
            Tool(
                name="colab_job_status",
                description="Check if background jobs are still running, their PIDs and elapsed time.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Specific job ID (optional, omit to see all jobs)",
                        },
                    },
                    "required": [],
                },
            ),
            Tool(
                name="colab_job_logs",
                description=(
                    "Get the last N lines of stdout from a background job. "
                    "Reads from a ring buffer (cheap, no disk I/O). "
                    "Use n_lines=20 for quick checks, n_lines=100 for debugging."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID to get logs from"},
                        "n_lines": {
                            "type": "integer",
                            "description": "Number of recent lines (default 50, max 200)",
                            "default": 50,
                        },
                    },
                    "required": ["job_id"],
                },
            ),
            Tool(
                name="colab_kill_job",
                description=(
                    "Kill a running background job. Sends SIGTERM then SIGKILL. "
                    "Use this to stop training before applying a fix."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job ID to kill"},
                    },
                    "required": ["job_id"],
                },
            ),

            # ── Command Execution ────────────────────────────────────
            Tool(
                name="colab_exec",
                description=(
                    "Run a SHORT shell command and wait for the result. "
                    "Output capped at 10KB to prevent token overflow. "
                    "For long-running commands, use colab_submit_job instead."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command"},
                        "timeout": {"type": "integer", "description": "Timeout seconds (max 600)", "default": 300},
                    },
                    "required": ["command"],
                },
            ),
            Tool(
                name="colab_python",
                description=(
                    "Execute a Python code snippet on Colab and return output. "
                    "For quick checks, introspection, and small tasks. "
                    "Output capped at 10KB."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "timeout": {"type": "integer", "description": "Timeout seconds (max 600)", "default": 300},
                    },
                    "required": ["code"],
                },
            ),

            # ── File Operations ──────────────────────────────────────
            Tool(
                name="colab_read_file",
                description=(
                    "Read a file on the Colab instance. "
                    "Capped at 50KB by default to prevent token overflow."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path on Colab"},
                        "max_bytes": {"type": "integer", "description": "Max bytes to read (default 50000)", "default": 50000},
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="colab_write_file",
                description=(
                    "Write content to a file on Colab. Creates parent directories. "
                    "Use this to write fixed training scripts, config files, etc."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path on Colab"},
                        "content": {"type": "string", "description": "File content to write"},
                    },
                    "required": ["path", "content"],
                },
            ),
            Tool(
                name="colab_upload",
                description="Upload a local file to the Colab instance (max 50MB).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "local_path": {"type": "string", "description": "Local file path"},
                        "remote_path": {"type": "string", "description": "Destination path on Colab"},
                    },
                    "required": ["local_path", "remote_path"],
                },
            ),
            Tool(
                name="colab_download",
                description="Download a file from Colab to local disk (max 50MB).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "remote_path": {"type": "string", "description": "File path on Colab"},
                        "local_path": {"type": "string", "description": "Local destination path"},
                    },
                    "required": ["remote_path", "local_path"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        handlers = {
            "colab_status": lambda a: _call(lambda c: c.health()),
            "colab_training_summary": lambda a: _call(lambda c: c.training_summary(last_n=a.get("last_n_entries", 50))),
            "colab_training_logs_raw": lambda a: _call(lambda c: c.training_logs_raw(last_n=a.get("last_n", 20))),
            "colab_list_checkpoints": lambda a: _call(lambda c: c.list_checkpoints(a.get("checkpoint_dir"))),
            "colab_submit_job": lambda a: _call(lambda c: c.submit_job(a["command"])),
            "colab_job_status": lambda a: _call(lambda c: c.job_status(a.get("job_id"))),
            "colab_job_logs": lambda a: _call(lambda c: c.job_logs(a["job_id"], a.get("n_lines", 50))),
            "colab_kill_job": lambda a: _call(lambda c: c.kill_job(a["job_id"])),
            "colab_exec": lambda a: _call(lambda c: c.exec(a["command"], a.get("timeout", 300))),
            "colab_python": lambda a: _call(lambda c: c.python(a["code"], a.get("timeout", 300))),
            "colab_read_file": lambda a: _call(lambda c: c.read_file(a["path"], a.get("max_bytes", 50000))),
            "colab_write_file": lambda a: _call(lambda c: c.write_file(a["path"], a["content"])),
            "colab_upload": lambda a: _call(lambda c: c.upload(a["local_path"], a["remote_path"])),
            "colab_download": lambda a: _call(lambda c: c.download(a["remote_path"], a["local_path"])),
        }

        handler = handlers.get(name)
        if handler is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        result = handler(arguments)
        return [TextContent(type="text", text=result)]

    return server


async def run_server():
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
