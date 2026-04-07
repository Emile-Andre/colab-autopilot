# Colab Autopilot — Agent Instructions

You have access to a remote Google Colab GPU via MCP tools. The Colab runtime has GPU, your training code, and model weights. Use these tools to monitor, debug, and fix training autonomously.

## Available Tools

| Tool | Purpose | Token Cost |
|------|---------|------------|
| `colab_status` | Check GPU, VRAM, disk, running jobs | Low |
| `colab_training_summary` | **USE THIS FIRST** — compact metrics digest with anomaly detection | Low |
| `colab_training_logs_raw` | Raw JSONL entries (use sparingly) | Medium |
| `colab_submit_job` | Launch long-running training (returns immediately) | Low |
| `colab_job_status` | Check if jobs are running | Low |
| `colab_job_logs` | Last N lines of job stdout (from ring buffer) | Medium |
| `colab_kill_job` | Stop a running job | Low |
| `colab_exec` | Run shell command (waits for result, output capped) | Medium |
| `colab_python` | Run Python snippet (waits for result, output capped) | Medium |
| `colab_read_file` | Read file on Colab (capped at 50KB) | Medium |
| `colab_write_file` | Write/edit file on Colab | Low |
| `colab_upload` / `colab_download` | Transfer files | High |
| `colab_list_checkpoints` | List saved model weights | Low |

## CRITICAL: Token Management Rules

1. **NEVER** stream raw training stdout into your context. Training output can be millions of lines.
2. **ALWAYS** use `colab_training_summary` first — it returns a compact digest (~500 tokens) with anomaly flags.
3. Only use `colab_job_logs` with `n_lines=20-50` for targeted debugging.
4. Only use `colab_training_logs_raw` with `last_n=10-20` when you need exact numerical values.
5. Output from `colab_exec` and `colab_python` is capped at 10KB server-side.

## Autonomous Monitoring Workflow

When asked to monitor training:

```
1. colab_training_summary           → Check metrics + anomalies
2. If anomalies == ["none"]:        → Report "training healthy" and WAIT
3. If anomaly detected:
   a. colab_job_logs(n_lines=30)    → Look at recent output for error context
   b. Diagnose the issue
   c. colab_kill_job(job_id)        → Stop the broken run
   d. colab_read_file(script_path)  → Read the code that needs fixing
   e. colab_write_file(script_path) → Write the fixed version
   f. colab_submit_job(command)     → Restart training
   g. Wait, then colab_training_summary → Verify the fix worked
```

## Autonomous Debug Loop

When training has a problem:

1. **Identify**: Use `colab_training_summary` to see what's wrong (NaN loss, reward collapse, etc.)
2. **Investigate**: Use `colab_job_logs(n_lines=50)` to see the error output
3. **Stop**: Use `colab_kill_job` to halt the broken run
4. **Read**: Use `colab_read_file` to read the training script
5. **Fix**: Use `colab_write_file` to write the corrected script
6. **Restart**: Use `colab_submit_job` to relaunch training
7. **Verify**: Wait ~2 minutes, then `colab_training_summary` to check the fix

## Key Paths (default, may vary)

- Training log: `/content/drive/MyDrive/world_model/training_logs.jsonl`
- Checkpoints: `/content/drive/MyDrive/world_model/weights/`
- Working dir: `/content/`

## Important Notes

- Colab sessions last max 12 hours. `colab_status` shows uptime.
- If the tunnel dies, the user needs to re-run the Colab cells and `colab-autopilot connect` again.
- Long-running jobs survive API restarts — they run in separate subprocesses.
- The training loop already writes structured JSONL logs. The summary endpoint parses those.
