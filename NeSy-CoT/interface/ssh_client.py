"""Thin paramiko wrapper for password-auth SSH/SFTP.

Plain OpenSSH (as used manually in README_VM.md) can't be driven headlessly
with a password on Windows without an interactive prompt. paramiko lets the
Streamlit app connect, upload/download, and start a detached tmux session
without ever writing the password to disk — it lives only in memory for the
duration of the Streamlit session (see the "always ask fresh" decision).
"""
from __future__ import annotations

import posixpath
import shlex
import stat
from pathlib import Path

import paramiko


class SSHError(RuntimeError):
    pass


class SSHClient:
    def __init__(self, host: str, username: str, password: str, port: int = 22, timeout: float = 15.0):
        self.host = host
        self.username = username
        self.port = port
        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self._client.connect(
                hostname=host,
                port=port,
                username=username,
                password=password,
                timeout=timeout,
                allow_agent=False,
                look_for_keys=False,
            )
        except Exception as exc:  # paramiko raises several distinct types
            raise SSHError(f"Could not connect to {username}@{host}:{port} — {exc}") from exc
        self._sftp = self._client.open_sftp()

    def close(self) -> None:
        try:
            self._sftp.close()
        finally:
            self._client.close()

    def __enter__(self) -> "SSHClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def run(self, command: str, timeout: float = 60.0) -> tuple[int, str, str]:
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        stdin.close()
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return exit_status, out, err

    def run_checked(self, command: str, timeout: float = 60.0) -> str:
        status, out, err = self.run(command, timeout=timeout)
        if status != 0:
            raise SSHError(f"Remote command failed ({status}): {command}\n{err or out}")
        return out

    # -- tmux-backed background execution ------------------------------------

    def tmux_session_exists(self, session_name: str) -> bool:
        status, _out, _err = self.run(f"tmux has-session -t {shlex.quote(session_name)} 2>/dev/null")
        return status == 0

    def start_background(self, session_name: str, remote_script_path: str) -> None:
        if self.tmux_session_exists(session_name):
            raise SSHError(f"A run is already active in tmux session '{session_name}'.")
        cmd = (
            f"tmux new-session -d -s {shlex.quote(session_name)} "
            f"{shlex.quote(f'bash {remote_script_path}')}"
        )
        self.run_checked(cmd)

    # -- filesystem -------------------------------------------------------------

    def mkdir_p(self, remote_dir: str) -> None:
        self.run_checked(f"mkdir -p {shlex.quote(remote_dir)}")

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        remote_dir = posixpath.dirname(remote_path)
        if remote_dir:
            self.mkdir_p(remote_dir)
        self._sftp.put(str(local_path), remote_path)

    def upload_text(self, text: str, remote_path: str) -> None:
        remote_dir = posixpath.dirname(remote_path)
        if remote_dir:
            self.mkdir_p(remote_dir)
        with self._sftp.open(remote_path, "w") as f:
            f.write(text)

    def upload_dir(self, local_dir: Path, remote_dir: str) -> int:
        """Recursively upload local_dir's contents into remote_dir. Returns file count."""
        self.mkdir_p(remote_dir)
        count = 0
        for path in sorted(local_dir.rglob("*")):
            rel = path.relative_to(local_dir).as_posix()
            remote_path = posixpath.join(remote_dir, rel)
            if path.is_dir():
                self.mkdir_p(remote_path)
            else:
                self.mkdir_p(posixpath.dirname(remote_path))
                self._sftp.put(str(path), remote_path)
                count += 1
        return count

    def download_file(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._sftp.get(remote_path, str(local_path))

    def read_remote_file(self, remote_path: str) -> str | None:
        try:
            with self._sftp.open(remote_path, "r") as f:
                return f.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return None

    def tail_remote_file(self, remote_path: str, lines: int = 200) -> str:
        status, out, err = self.run(f"tail -n {int(lines)} {shlex.quote(remote_path)} 2>/dev/null")
        if status != 0:
            return ""
        return out

    def remote_exists(self, remote_path: str) -> bool:
        try:
            self._sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False

    def is_remote_dir(self, remote_path: str) -> bool:
        try:
            return stat.S_ISDIR(self._sftp.stat(remote_path).st_mode)
        except FileNotFoundError:
            return False
