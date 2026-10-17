"""SSH/SFTP control with a password supplied through stdin, never saved to disk.

After authentication, accepts one JSON operation per stdin line. Intended for
remote setup, monitoring and result transfer; the training job runs separately.
"""
import argparse
import hashlib
import getpass
import json
from pathlib import Path
import sys
import time

import paramiko


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="riftuser")
    args = parser.parse_args()
    print("Supply SSH password through stdin; it will remain in process memory only.", flush=True)
    password = (getpass.getpass("SSH password: ") if sys.stdin.isatty()
                else sys.stdin.readline().rstrip("\r\n"))
    if not password:
        raise ValueError("No password was supplied; use an interactive terminal")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def connect():
        client.connect(args.host, username=args.user, password=password,
                       timeout=25, auth_timeout=25, banner_timeout=25,
                       allow_agent=False, look_for_keys=False)
        client.get_transport().set_keepalive(30)

    connect()
    fingerprint = hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()
    print(json.dumps({"connected": args.host, "host_key_sha256_hex": fingerprint}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not client.get_transport() or not client.get_transport().is_active():
                connect()
            action = request["action"]
            if action == "exec":
                channel = client.get_transport().open_session()
                channel.exec_command(request["command"])
                stdout, stderr = [], []
                deadline = time.monotonic() + request.get("timeout", 45)
                while True:
                    while channel.recv_ready():
                        stdout.append(channel.recv(65536))
                    while channel.recv_stderr_ready():
                        stderr.append(channel.recv_stderr(65536))
                    if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Remote control command exceeded its timeout")
                    time.sleep(0.1)
                result = {"exit_code": channel.recv_exit_status(),
                          "stdout": b"".join(stdout).decode("utf-8", errors="replace"),
                          "stderr": b"".join(stderr).decode("utf-8", errors="replace")}
                channel.close()
            elif action in ("upload", "download"):
                with client.open_sftp() as sftp:
                    if action == "upload":
                        sftp.put(request["local"], request["remote"])
                    else:
                        local = Path(request["local"])
                        local.parent.mkdir(parents=True, exist_ok=True)
                        sftp.get(request["remote"], str(local))
                result = {"action": action, "local": request["local"], "remote": request["remote"]}
            elif action == "close":
                break
            else:
                raise ValueError("Unknown operation")
            print(json.dumps(result), flush=True)
        except Exception as error:
            print(json.dumps({"error_type": type(error).__name__, "error": str(error)}), flush=True)
    client.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
