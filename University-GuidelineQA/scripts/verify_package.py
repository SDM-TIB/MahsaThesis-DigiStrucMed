"""Verify this handover's files with Python's standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    manifest = json.loads((ROOT / "SHA256SUMS.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["files"].items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            failures.append(name + ": missing or invalid path")
            continue
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            failures.append(name + ": checksum mismatch")
    if failures:
        print("Integrity check failed:\n" + "\n".join(failures))
        raise SystemExit(1)
    print(f"All {len(manifest['files'])} delivered files verified.")
    from verify_gaapo_prompt import verify
    audit = verify(ROOT / "web/gaapo")
    print(f"GAAPO best prompt verified: {audit['candidate_id']}.")


if __name__ == "__main__":
    main()
