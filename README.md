# colab-autopilot

Give Claude Code (or any MCP-compatible AI agent) autonomous access to your Google Colab GPU — no browser tab needed after setup.

Built for long-running RL training workflows where you want the agent to monitor, detect issues, stop, debug, fix, and restart training while you're away.

## How It Works

```
┌─────────────────┐     cloudflared tunnel (E2E encrypted)     ┌──────────────────┐
│  Your Machine   │ ──────── HTTPS ─────────────────────────► │  Google Colab     │
│                 │                                            │  (GPU Runtime)    │
│  Claude Code    │ ◄── compact metrics summary ────────────── │  Flask API        │
│  ↕ MCP Server   │ ──── kill job / write fix ────────────────► │  Job Manager      │
│                 │ ◄── job status ─────────────────────────── │  Training Loop    │
└─────────────────┘                                            └──────────────────┘

No browser tab needed after initial setup.
No Google Drive sync delays.
Sub-second round-trip communication.
```

## Key Features

- **No browser tab needed** — Colab runs headless after you start the tunnel
- **Token-safe** — Training output never floods the agent's context. A `/training_summary` endpoint returns ~500 tokens with metrics digest + anomaly detection
- **Background jobs** — Submit training via `/submit_job`, monitor with `/job_logs`, kill with `/kill_job`
- **Autonomous debug loop** — Agent can: detect NaN/reward collapse → stop run → read code → write fix → restart
- **E2E encrypted** — Fernet AES-128 encryption. Cloudflare only sees ciphertext
- **12 MCP tools** — Status, training summary, raw logs, exec, python, submit/kill/monitor jobs, file read/write, upload/download, checkpoints

## Setup

### 1. Install locally (one time)

```bash
pip install git+https://github.com/Emile-Andre/colab-autopilot.git
```

### 2. Add to Claude Code config

Edit `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "colab-autopilot": {
      "command": "colab-autopilot",
      "args": ["mcp-serve"]
    }
  }
}
```

### 3. Set up Colab (each new session)

Copy the 3 setup cells from `notebooks/autopilot_setup.ipynb` into the TOP of your Colab notebook, before any model code. Or just paste them manually:

**Cell 1:** Install deps
```python
!pip install -q flask cryptography
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared 2>/dev/null
!chmod +x /usr/local/bin/cloudflared
```

**Cell 2:** Start server (see `notebooks/colab_autopilot_server.py`)

**Cell 3:** Open tunnel — prints the `cc://...` connection string

### 4. Connect (each new session)

```bash
colab-autopilot connect cc://TOKEN:KEY@host
```

### 5. Close the browser tab

Your Colab runtime keeps running. Claude Code now has full GPU access.

## Configuration

Edit these variables in the Cell 2 server code to match your notebook:

```python
TRAINING_LOG_PATH = "/content/drive/MyDrive/world_model/training_logs.jsonl"
CHECKPOINT_DIR = "/content/drive/MyDrive/world_model/weights"
WORK_DIR = "/content"
```

## Token Budget

The system is designed to keep agent context usage minimal:

| Operation | ~Tokens |
|-----------|---------|
| `colab_training_summary` | 300-500 |
| `colab_job_logs(n_lines=20)` | 200-400 |
| `colab_status` | 100-200 |
| `colab_exec` (small output) | 200-500 |
| Full training stdout (raw) | 50,000+ ❌ |

The summary endpoint does the heavy lifting server-side: it reads your JSONL logs, computes rolling averages, samples the loss curve to 10 points, and runs anomaly detection — all before sending anything to the agent.

## Compared to ColabWatcher4aiAgents

| Feature | ColabWatcher | colab-autopilot |
|---------|-------------|-----------------|
| Transport | Google Drive sync | Cloudflare tunnel (HTTPS) |
| Latency | 15-60+ seconds | Sub-second |
| Browser needed | No (Drive-based) | No (tunnel-based) |
| Real-time logs | No (file-based) | Yes (ring buffer) |
| Token management | No | Yes (summary endpoint) |
| Background jobs | Sequential queue | Parallel with kill/monitor |
| Anomaly detection | No | Yes (NaN, collapse, stagnation) |
| File editing | Via Drive | Direct API |
| Encryption | No | Fernet E2E |

## License

MIT
