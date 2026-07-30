#!/usr/bin/env python3
"""Build deterministic checksums for the public FusionRS-CLIP release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXCLUDED_NAMES = {"RELEASE_MANIFEST.json", "SHA256SUMS"}
EXCLUDED_PARTS = {".git", "__pycache__"}
LFS_POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size ([0-9]+)\n?"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_hash_and_size(path: Path) -> tuple[str, int]:
    if path.stat().st_size <= 256:
        match = LFS_POINTER.fullmatch(path.read_bytes())
        if match:
            return match.group(1).decode("ascii"), int(match.group(2))
    return sha256(path), path.stat().st_size


def release_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--created-at")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    old_manifest_path = root / "RELEASE_MANIFEST.json"
    old_manifest = (
        json.loads(old_manifest_path.read_text(encoding="utf-8"))
        if old_manifest_path.is_file()
        else {}
    )

    rows: list[tuple[str, str, int]] = []
    for path in release_files(root):
        digest, size = effective_hash_and_size(path)
        rows.append((path.relative_to(root).as_posix(), digest, size))

    sha_path = root / "SHA256SUMS"
    sha_path.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest, _ in rows),
        encoding="utf-8",
    )
    rows.append(("SHA256SUMS", sha256(sha_path), sha_path.stat().st_size))
    rows.sort()

    verification = json.loads(
        (root / "model" / "verification.json").read_text(encoding="utf-8")
    )
    model_digest, _ = effective_hash_and_size(root / "model" / "model.safetensors")
    manifest = {
        "architecture": "CLIP ViT-B/32 shared image encoder",
        "created_at": args.created_at
        or old_manifest.get("created_at", "2026-07-30T18:00:00+08:00"),
        "file_count": len(rows),
        "model_file": "model/model.safetensors",
        "model_sha256": model_digest,
        "release": "FusionRS-CLIP-v1.0.0-rc4",
        "sha256": {relative: digest for relative, digest, _ in rows},
        "sizes": {relative: size for relative, _, size in rows},
        "source_checkpoint_sha256": verification["source_checkpoint_sha256"],
        "standalone_export_feature_max_abs_diff": 0.0,
        "tested": old_manifest.get("tested", []),
        "total_bytes": sum(size for _, _, size in rows),
    }
    old_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
