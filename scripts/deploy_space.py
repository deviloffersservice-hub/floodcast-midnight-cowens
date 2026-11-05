"""Deploy both apps (HTML/CSS app at /, Gradio app at /gradio/) to a Hugging Face Space.

The Space uses the free Gradio SDK: it runs space_app.py, which starts server.py on port 7860
(packages.txt adds libgomp1 for LightGBM). The Dockerfile is for other container hosts.

Only the files the apps need at runtime are uploaded (no training CSVs). The Hugging Face upload
API handles binary files (parquet), which a plain `git push` to a Space would reject.

Usage:
  .venv/bin/hf auth login                                   # once; token with "Write" access
  .venv/bin/python scripts/deploy_space.py USER/SPACE_NAME  # creates the Space if needed, uploads
  .venv/bin/python scripts/deploy_space.py --bundle-only DIR    # just assemble the files (for testing)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_README = ROOT / "deploy" / "SPACE_README.md"
# Everything the two apps read at runtime (see app.py, server.py, floodcast/forecast.py).
RUNTIME_PATHS = [
    "space_app.py", "packages.txt", "requirements.txt", "app.py", "server.py", "solution.py",
    "floodcast", "models", "web", "judge_test", "reports/replay_events.json",
    "data/snapshot", "data/processed/full_daily.parquet",
]
KEEP_REMOTE = {".gitattributes", "README.md"}
# A Gradio-SDK Space also installs gradio[oauth,mcp]; its "mcp" extra needs pydantic<=2.12.5, which
# clashes with the laptop pin. Only these pins are relaxed in the Space copy of requirements.txt; the
# model packages (lightgbm, numpy, pandas, scikit-learn, ...) keep their exact versions.
SPACE_UNPINNED = ("pydantic", "pydantic-core")
HF_FILE_LIMIT_MB = 50  # keep every file small; the biggest runtime file is ~9 MB


def tracked_files() -> list[str]:
    """Runtime files as committed in git (never stray local files such as caches)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", *RUNTIME_PATHS], cwd=ROOT, check=True,
                         capture_output=True).stdout.decode()
    files = [f for f in out.split("\0") if f]
    missing = [p for p in RUNTIME_PATHS if not any(f == p or f.startswith(p + "/") for f in files)]
    if missing:
        raise SystemExit(f"not tracked in git (commit them first): {missing}")
    return files


def assemble(out_dir: Path) -> list[str]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty")
    files = tracked_files()
    for rel in files:
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
        size_mb = dest.stat().st_size / 1e6
        if size_mb > HF_FILE_LIMIT_MB:
            raise SystemExit(f"{rel} is {size_mb:.0f} MB, too large for the Space")
    reqs = (out_dir / "requirements.txt").read_text(encoding="utf-8").splitlines()
    relaxed = [line.split("==")[0] if line.split("==")[0].strip().lower() in SPACE_UNPINNED else line
               for line in reqs]
    (out_dir / "requirements.txt").write_text("\n".join(relaxed) + "\n", encoding="utf-8")
    shutil.copy2(SPACE_README, out_dir / "README.md")  # the Space card (sdk: gradio, app_file: space_app.py)
    return [*files, "README.md"]


def deploy(space: str, bundle: Path, files: list[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001 - any auth failure means "log in first"
        raise SystemExit(f"Not logged in to Hugging Face ({exc}). Run: .venv/bin/hf auth login") from exc
    print(f"Logged in as {user}; deploying to Space {space}")
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    api.upload_folder(folder_path=str(bundle), repo_id=space, repo_type="space",
                      commit_message=f"Deploy FLOODCAST from GitHub {sha}")
    stale = sorted(set(api.list_repo_files(space, repo_type="space")) - set(files) - KEEP_REMOTE)
    if stale:
        api.delete_files(space, delete_patterns=stale, repo_type="space",
                         commit_message="Remove files no longer deployed")
    print(f"Uploaded {len(files)} files. Build log and app: https://huggingface.co/spaces/{space}")
    print(f"App URL once built: https://{space.replace('/', '-').replace('_', '-').lower()}.hf.space/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", nargs="?", help="USER/SPACE_NAME on huggingface.co")
    parser.add_argument("--bundle-only", type=Path, help="only copy the deploy files into this empty folder")
    args = parser.parse_args()
    if args.bundle_only:
        files = assemble(args.bundle_only)
        print(f"Assembled {len(files)} files in {args.bundle_only}")
        return
    if not args.space or "/" not in args.space:
        parser.error("give the Space as USER/SPACE_NAME, e.g. midnight-cowens/floodcast")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        deploy(args.space, bundle, assemble(bundle))


if __name__ == "__main__":
    sys.exit(main())
