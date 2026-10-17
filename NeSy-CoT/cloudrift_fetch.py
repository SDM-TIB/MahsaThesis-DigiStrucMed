"""Wait for a detached VM run, then download and verify its results.

Usage: python cloudrift_fetch.py --run outputs/<run>/remote-run.json --wait
The SSH password is prompted securely and is never saved in metadata or logs.
"""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

import paramiko


class PinnedHostKey(paramiko.MissingHostKeyPolicy):
    def __init__(self, expected):
        self.expected = expected

    def missing_host_key(self, client, hostname, key):
        actual = hashlib.sha256(key.asbytes()).hexdigest()
        if actual != self.expected:
            raise paramiko.SSHException("VM host key differs from the key recorded during setup")
        client.get_host_keys().add(hostname, key.get_name(), key)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    metadata_path = Path(args.run).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    local = metadata_path.parent
    password = getpass.getpass("SSH password: ")
    remote_root = metadata["remote_root"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(PinnedHostKey(metadata["host_key_sha256_hex"]))
    def connect():
        client.connect(metadata["host"], username=metadata["user"], password=password,
                       timeout=25, banner_timeout=25, auth_timeout=25,
                       allow_agent=False, look_for_keys=False)
        client.get_transport().set_keepalive(30)
    connect()
    while True:
        try:
            if not client.get_transport() or not client.get_transport().is_active():
                connect()
            with client.open_sftp() as sftp:
                with sftp.open(remote_root + "/status.txt", "r") as stream:
                    status = stream.read().decode("utf-8").strip()
                with sftp.open(remote_root + "/logs/pipeline.log", "r") as stream:
                    size = stream.stat().st_size
                    stream.seek(max(0, size - 1200))
                    lines = stream.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
                    recent = next((line.strip() for line in reversed(lines) if line.strip()), "")
            summary = {"status": status, "progress": recent[-250:]}
            message = json.dumps(summary, ensure_ascii=True)
            print(message, flush=True)
            with (local / "monitor.log").open("a", encoding="utf-8") as stream:
                stream.write(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()) + " " + message + "\n")
            if status == "SUCCESS":
                break
            if status.startswith("FAILED"):
                with client.open_sftp() as sftp:
                    sftp.get(remote_root + "/logs/pipeline.log", str(local / "pipeline.failed.log"))
                raise RuntimeError("VM pipeline failed; diagnostic log downloaded")
            if not args.wait:
                print("The run is still active. Use --wait or retry after completion.", flush=True)
                return
        except (OSError, EOFError, paramiko.SSHException) as error:
            if not args.wait:
                raise
            print(f"Connection interrupted ({type(error).__name__}); retrying.", flush=True)
            client.close()
        time.sleep(45)
    archive = local / "results.tar.gz"
    partial = local / "results.tar.gz.partial"
    with client.open_sftp() as sftp:
        with sftp.open(remote_root + "/results.tar.gz.sha256", "r") as stream:
            expected = stream.read().decode("utf-8").split()[0]
        print("Downloading final adapter, evaluation results and logs...", flush=True)
        sftp.get(remote_root + "/results.tar.gz", str(partial))
        sftp.get(remote_root + "/logs/pipeline.log", str(local / "pipeline.log"))
    if digest(partial) != expected:
        raise ValueError("Downloaded archive failed SHA-256 verification")
    partial.replace(archive)
    extracted = local / "downloaded-results"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as package:
        package.extractall(extracted, filter="data")
    result_root = extracted / "NeSy-CoT/outputs/T4B_cloudrift"
    manifest = json.loads((result_root / "result_manifest.json").read_text())
    for relative, expected_hash in manifest.items():
        if digest(result_root / relative) != expected_hash:
            raise ValueError(f"Result-file checksum mismatch: {relative}")
    results = json.loads((result_root / "finetune_with_rules_CoT2_results.json").read_text())
    metadata["download_complete"] = True
    metadata["results_directory"] = str(result_root)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"download_complete": True, "results_directory": str(result_root),
                      "metrics": results.get("metrics", {})}, ensure_ascii=True), flush=True)
    client.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
