"""Audit the bundled prompt against the completed GAAPO run, without ML packages."""
import hashlib
import json
from pathlib import Path


def verify(directory):
    artifact = json.loads((directory / "prompt.json").read_text(encoding="utf-8"))
    raw = (directory / "best_prompt.txt").read_bytes()
    prompt = raw.decode("utf-8").strip()
    candidates = [json.loads(line) for line in (directory / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    ranked = sorted((c for c in candidates if c.get("evaluation_score") is not None),
                    key=lambda c: (-c["evaluation_score"], c["id"]))
    comparison = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
    experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    provenance = artifact["provenance"]
    checks = {
        "completed real run": comparison.get("dry_run") is False and comparison.get("smoke_test") is False,
        "highest-scoring selected candidate": bool(ranked) and ranked[0]["prompt_text"].strip() == prompt,
        "candidate provenance": bool(ranked) and ranked[0]["id"] == provenance["candidate_id"]
            and ranked[0]["evaluation_score"] == provenance["validation_score"]
            and ranked[0]["generation_method"] == provenance["generation_method"],
        "exact best_prompt file": hashlib.sha256(raw).hexdigest() == provenance["original_file_sha256"],
        "normalized prompt integrity": artifact["prompt_text"] == prompt
            and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == provenance["sha256"],
        "system message integrity": artifact["schema_version"] == 1
            and artifact["system_prompt"] == prompt + "\nOutput requirements: " + artifact["output_requirements"]
            and hashlib.sha256(artifact["system_prompt"].encode("utf-8")).hexdigest() == provenance["system_prompt_sha256"],
        "original evaluation context": provenance["evaluation_objective"] == experiment["storyboard_evaluation"]
            and provenance["original_test_comparison"] == comparison,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("GAAPO prompt audit failed: " + ", ".join(failed))
    return {"candidate_id": provenance["candidate_id"], "sha256": provenance["sha256"], "passed": list(checks)}


if __name__ == "__main__":
    result = verify(Path(__file__).resolve().parent.parent / "web/gaapo")
    print(json.dumps(result, indent=2))
