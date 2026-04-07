"""HTTP client with E2E encryption for colab-autopilot.

All endpoints except /health use Fernet-encrypted payloads.
Output is always capped to prevent token explosion in the agent.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Union

import httpx

from colab_autopilot.crypto import decrypt, encrypt

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
DEFAULT_TIMEOUT = 300
MAX_TIMEOUT = 600


class ColabError(Exception):
    """Error communicating with the Colab runtime."""


class ColabClient:
    def __init__(self, url: str, token: str, encryption_key: Union[bytes, str]):
        self.url = url.rstrip("/")
        self.token = token
        self.key = encryption_key.encode("utf-8") if isinstance(encryption_key, str) else encryption_key
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(MAX_TIMEOUT + 30),
        )

    def _post_encrypted(self, endpoint: str, data: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
        data["timeout"] = min(timeout, MAX_TIMEOUT)
        ciphertext = encrypt(self.key, data)
        try:
            resp = self._http.post(
                f"{self.url}{endpoint}",
                content=ciphertext,
                headers={"Content-Type": "application/octet-stream"},
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise ColabError("Colab session may have timed out. Restart the notebook and reconnect.")
        except httpx.ReadTimeout:
            raise ColabError(f"Command timed out after {timeout}s.")
        if resp.status_code == 401:
            raise ColabError("Invalid token.")
        resp.raise_for_status()
        return decrypt(self.key, resp.content)

    def health(self) -> dict:
        try:
            resp = self._http.get(f"{self.url}/health")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise ColabError("Colab session may have timed out.")
        resp.raise_for_status()
        return resp.json()

    def exec(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
        return self._post_encrypted("/exec", {"command": command}, timeout)

    def python(self, code: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
        return self._post_encrypted("/python", {"code": code}, timeout)

    def submit_job(self, command: str) -> dict:
        """Submit a long-running job. Returns immediately with job_id."""
        return self._post_encrypted("/submit_job", {"command": command})

    def job_status(self, job_id: str = None) -> dict:
        return self._post_encrypted("/job_status", {"job_id": job_id})

    def job_logs(self, job_id: str, n_lines: int = 50) -> dict:
        return self._post_encrypted("/job_logs", {"job_id": job_id, "n_lines": n_lines})

    def kill_job(self, job_id: str) -> dict:
        return self._post_encrypted("/kill_job", {"job_id": job_id})

    def training_summary(self, log_path: str = None, last_n: int = 50) -> dict:
        data = {"last_n_entries": last_n}
        if log_path:
            data["log_path"] = log_path
        return self._post_encrypted("/training_summary", data)

    def training_logs_raw(self, log_path: str = None, last_n: int = 20) -> dict:
        data = {"last_n": last_n}
        if log_path:
            data["log_path"] = log_path
        return self._post_encrypted("/training_logs_raw", data)

    def list_checkpoints(self, checkpoint_dir: str = None) -> dict:
        data = {}
        if checkpoint_dir:
            data["checkpoint_dir"] = checkpoint_dir
        return self._post_encrypted("/list_checkpoints", data)

    def read_file(self, path: str, max_bytes: int = 50000) -> dict:
        return self._post_encrypted("/read_file", {"path": path, "max_bytes": max_bytes})

    def write_file(self, path: str, content: str) -> dict:
        return self._post_encrypted("/write_file", {"path": path, "content": content})

    def upload(self, local_path: str, remote_path: str) -> dict:
        path = Path(local_path)
        if not path.exists():
            raise ColabError(f"Local file not found: {local_path}")
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            raise ColabError(f"File exceeds 50MB limit ({size / 1024 / 1024:.1f}MB).")
        content_b64 = base64.b64encode(path.read_bytes()).decode()
        return self._post_encrypted("/upload", {"path": remote_path, "content_b64": content_b64})

    def download(self, remote_path: str, local_path: str) -> dict:
        result = self._post_encrypted("/download", {"path": remote_path})
        content = base64.b64decode(result["content_b64"])
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        return result

    def close(self) -> None:
        self._http.close()
