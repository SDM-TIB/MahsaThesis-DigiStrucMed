"""Optional explicit deployment of this model to a NEW Hugging Face ZeroGPU Space.

Requires huggingface-hub==0.36.2. Default visibility is private; --public opts in
to publishing the app, adapter and source. No token is written to a local file.
"""
import argparse
import getpass
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Your HF username/Llama-GuidelineKG-QA")
    parser.add_argument("--public", action="store_true", help="Publish the Space and its source/adapter")
    args = parser.parse_args()
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.utils import HfHubHTTPError

    token = os.environ.get("HF_TOKEN") or get_token() or getpass.getpass("Hugging Face token (not saved locally): ")
    if not token:
        raise SystemExit("An HF token is required for this optional upload. Local CPU Q&A needs no token.")
    api = HfApi(token=token)
    try:
        api.whoami()
        # A new repo is mandatory; never silently overwrite an existing Space.
        api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="gradio",
                        space_hardware="zero-a10g", private=not args.public, exist_ok=False)
        api.add_space_secret(repo_id=args.repo_id, key="HF_TOKEN", value=token)
        with tempfile.TemporaryDirectory(prefix="guidelineqa-space-") as temporary:
            directory = Path(temporary)
            for name in ("app.py", "requirements.txt", "README.md"):
                shutil.copy2(ROOT / "huggingface" / name, directory / name)
            shutil.copytree(ROOT / "adapter", directory / "adapter")
            shutil.copytree(ROOT / "licenses", directory / "licenses")
            shutil.copytree(ROOT / "web/gaapo", directory / "gaapo")
            api.upload_folder(repo_id=args.repo_id, repo_type="space", folder_path=directory,
                              commit_message="Add standalone GAAPO best-prompt and GuidelineKG Q&A demonstration")
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise SystemExit(f"HF deployment could not finish (HTTP {status}). Check token write permissions, a NEW repo name, gated base-model access, and ZeroGPU eligibility. No paid GPU is requested. The local CPU package remains available.") from None
    print(f"Uploaded: https://huggingface.co/spaces/{args.repo_id}")
    print("Wait for the build, then open the Space. Private Spaces require permitted HF accounts.")


if __name__ == "__main__":
    main()
