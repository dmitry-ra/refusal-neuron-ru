"""Download the pinned models into $HF_HOME and verify sha256 against the Hub LFS metadata."""

import argparse
import hashlib
import json
import os
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from snsteer import config  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(api, repo_id, revision, files=None):
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    expected = {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings}
    if files is None:
        root = snapshot_download(repo_id, revision=revision)
        names = sorted(expected)
    else:
        root = None
        names = files
        for name in files:
            p = hf_hub_download(repo_id, name, revision=revision)
            root = os.path.dirname(p)
    out = {}
    for name in names:
        path = os.path.join(root, name)
        if not os.path.exists(path):
            raise SystemExit(f"missing after download: {repo_id}@{revision}/{name}")
        got = sha256(path)
        want = expected.get(name)
        if want and want != got:
            raise SystemExit(f"sha256 mismatch: {repo_id}@{revision}/{name}: {got} != {want}")
        out[name] = {"sha256": got, "bytes": os.path.getsize(path), "lfs_verified": bool(want)}
    return {"repo_id": repo_id, "revision": revision, "path": root, "files": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="manifest JSON path")
    ap.add_argument("--which", default="model,guard,cvec")
    args = ap.parse_args()
    api = HfApi()
    which = args.which.split(",")
    manifest = []
    if "model" in which:
        manifest.append(fetch(api, config.MODEL_ID, config.MODEL_REVISION))
    if "guard" in which:
        manifest.append(fetch(api, config.GUARD_ID, config.GUARD_REVISION))
    if "cvec" in which:
        manifest.append(fetch(api, config.CVEC_ID, config.CVEC_REVISION, config.CVEC_FILES))
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=1)
    for m in manifest:
        for name, v in m["files"].items():
            print(f"{m['repo_id']}@{m['revision'][:8]} {name} {v['sha256']} {v['bytes']}")


if __name__ == "__main__":
    main()
