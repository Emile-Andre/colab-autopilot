"""Config management — stores connection details in ~/.colab-autopilot.json."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


def config_path() -> Path:
    return Path.home() / ".colab-autopilot.json"


def parse_uri(uri: str) -> Tuple[str, str, str]:
    """Parse cc://TOKEN:KEY@host into (url, token, encryption_key)."""
    if not uri.startswith("cc://"):
        raise ValueError("Connection string must start with cc://")
    
    rest = uri[5:]
    if "@" not in rest:
        raise ValueError("Missing @host in connection string")
    
    credentials, host = rest.rsplit("@", 1)
    if not host:
        raise ValueError("Empty host")
    
    if ":" not in credentials:
        raise ValueError("Must be cc://TOKEN:KEY@host")
    
    token, key = credentials.split(":", 1)
    if not token or not key:
        raise ValueError("Empty token or key")
    
    url = f"https://{host}"
    return url, token, key


def save_config(url: str, token: str, encryption_key: str) -> Path:
    path = config_path()
    data = {
        "url": url,
        "token": token,
        "encryption_key": encryption_key,
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2))
    os.chmod(path, 0o600)
    return path


def load_config() -> Optional[dict]:
    path = config_path()
    if not path.exists():
        return None
    return json.loads(path.read_text())


def clear_config() -> None:
    path = config_path()
    if path.exists():
        path.unlink()
