"""colab-autopilot CLI — connect, status, and MCP server."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import click

from colab_autopilot import __version__
from colab_autopilot.client import ColabClient, ColabError
from colab_autopilot.config import clear_config, load_config, parse_uri, save_config


def _require_client() -> ColabClient:
    config = load_config()
    if config is None:
        click.echo("Error: Not connected. Run: colab-autopilot connect cc://TOKEN:KEY@host", err=True)
        raise SystemExit(1)
    return ColabClient(url=config["url"], token=config["token"], encryption_key=config["encryption_key"])


@click.group()
@click.version_option(__version__)
def main():
    """colab-autopilot — Autonomous Colab GPU access for AI coding agents."""


@main.command()
@click.argument("uri", required=False)
def connect(uri: Optional[str]):
    """Save Colab connection. URI: cc://TOKEN:KEY@host"""
    if uri is None:
        uri = click.prompt("Connection string")
    elif uri == "-":
        uri = sys.stdin.readline().strip()

    try:
        url, token, key = parse_uri(uri)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    # Verify connection
    client = ColabClient(url=url, token=token, encryption_key=key)
    try:
        health = client.health()
        click.echo(f"Connected to {url}")
        click.echo(f"  GPU: {health.get('gpu', {}).get('name', 'unknown')}")
        click.echo(f"  VRAM: {health.get('gpu', {}).get('vram_total_gb', '?')}GB")
        click.echo(f"  Uptime: {health.get('uptime_seconds', '?')}s")
    except ColabError:
        click.echo(f"Warning: Connected but couldn't verify. Runtime may not be ready yet.")

    save_config(url, token, key)
    click.echo("Config saved. Claude Code can now use the MCP tools.")


@main.command()
def disconnect():
    """Clear saved connection."""
    clear_config()
    click.echo("Disconnected.")


@main.command()
def status():
    """Show GPU status and active jobs."""
    client = _require_client()
    try:
        info = client.health()
    except ColabError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    gpu = info.get("gpu", {})
    click.echo(f"GPU:     {gpu.get('name', 'unknown')}")
    click.echo(f"VRAM:    {gpu.get('vram_used_gb', '?')}/{gpu.get('vram_total_gb', '?')} GB")
    click.echo(f"Disk:    {info.get('disk_free', '?')} free")
    click.echo(f"Uptime:  {info.get('uptime_seconds', '?')}s")

    jobs = info.get("active_jobs", [])
    if jobs:
        click.echo(f"\nActive Jobs ({len(jobs)}):")
        for j in jobs:
            status = "RUNNING" if j["running"] else "FINISHED"
            click.echo(f"  [{j['job_id']}] {status} — {j['command']}")


@main.command()
def summary():
    """Show training metrics summary."""
    client = _require_client()
    try:
        result = client.training_summary()
        click.echo(json.dumps(result, indent=2))
    except ColabError as e:
        click.echo(f"Error: {e}", err=True)


@main.command("mcp-serve")
def mcp_serve():
    """Start the MCP server (stdio transport for Claude Code)."""
    from colab_autopilot.mcp_server import run_server
    asyncio.run(run_server())


import json  # noqa: E402 (needed for summary command)
