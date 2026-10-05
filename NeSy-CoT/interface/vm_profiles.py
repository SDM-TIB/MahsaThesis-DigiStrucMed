"""VM hardware profiles (CloudRift GPU rental SKUs).

Profiles hold only hardware specs and derived fine-tuning knobs (currently
just ``gpu_max_memory_mb``) — never credentials. IP/user/password are always
entered fresh per session (see interface/app.py), since CloudRift assigns a
new IP to each rented instance anyway.
"""
from __future__ import annotations

import json
from pathlib import Path

PROFILES_PATH = Path(__file__).resolve().parent / "vm_profiles.json"

_SEED_PROFILES = [
    {
        "id": "v100-sxm3-800",
        "label": "V100 SXM3 — ustx1a_a01 (800GB disk)",
        "gpu": "V100 SXM3",
        "vram_gb": 32,
        "ram_gb": 85,
        "disk_gb": 800,
        "price_per_hr": 0.28,
        "location": "ustx1a_a01 - USA",
        "gpu_max_memory_mb": 28000,
        "verified": False,
        "is_default": True,
    },
    {
        "id": "v100-sxm3-400",
        "label": "V100 SXM3 — usny01_a01 (400GB disk)",
        "gpu": "V100 SXM3",
        "vram_gb": 32,
        "ram_gb": 85,
        "disk_gb": 400,
        "price_per_hr": 0.28,
        "location": "usny01_a01 - USA",
        "gpu_max_memory_mb": 28000,
        "verified": False,
        "is_default": False,
    },
    {
        "id": "v100-sxm2-400",
        "label": "V100 SXM2 — usny01_a01 (400GB disk)",
        "gpu": "V100 SXM2",
        "vram_gb": 16,
        "ram_gb": 52,
        "disk_gb": 400,
        "price_per_hr": 0.25,
        "location": "usny01_a01 - USA",
        "gpu_max_memory_mb": 14000,
        "verified": True,
        "is_default": False,
    },
]


def default_profiles() -> list[dict]:
    return json.loads(json.dumps(_SEED_PROFILES))


def load_profiles(path: Path = PROFILES_PATH) -> list[dict]:
    if not path.exists():
        profiles = default_profiles()
        save_profiles(profiles, path)
        return profiles
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = data.get("profiles", [])
        if not profiles:
            raise ValueError("empty profiles list")
        return profiles
    except (json.JSONDecodeError, ValueError):
        return default_profiles()


def save_profiles(profiles: list[dict], path: Path = PROFILES_PATH) -> None:
    path.write_text(json.dumps({"profiles": profiles}, indent=2) + "\n", encoding="utf-8")


def get_profile(profiles: list[dict], profile_id: str) -> dict | None:
    return next((p for p in profiles if p["id"] == profile_id), None)


def default_profile(profiles: list[dict]) -> dict:
    return next((p for p in profiles if p.get("is_default")), profiles[0])
