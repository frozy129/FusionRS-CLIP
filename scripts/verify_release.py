from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a FusionRS release")
    parser.add_argument("--root", default=".", help="Release root")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    manifest_path = root / "RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    expected_files = set(manifest["sha256"]) | {"RELEASE_MANIFEST.json"}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (
            path.is_file()
            and ".git" not in path.relative_to(root).parts
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix != ".pyc"
        )
    }
    for relative in sorted(actual_files - expected_files):
        errors.append(f"unexpected: {relative}")

    for relative, expected in manifest["sha256"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(f"hash mismatch: {relative}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    from fusionrs import FusionRSEncoder

    encoder = FusionRSEncoder.from_pretrained(root / "model", device="cpu")
    rgb = Image.new("RGB", (224, 224), (20, 100, 180))
    ir = Image.new("L", (224, 224), 140)
    image_features = encoder.encode_images([rgb, ir])
    text_features = encoder.encode_text(["a remote sensing scene"])
    if tuple(image_features.shape) != (2, 512):
        raise RuntimeError(f"Unexpected image shape: {tuple(image_features.shape)}")
    if tuple(text_features.shape) != (1, 512):
        raise RuntimeError(f"Unexpected text shape: {tuple(text_features.shape)}")
    if not torch.isfinite(image_features).all() or not torch.isfinite(text_features).all():
        raise RuntimeError("Non-finite features")

    print(
        json.dumps(
            {
                "status": "ok",
                "files_verified": len(manifest["sha256"]),
                "image_shape": list(image_features.shape),
                "text_shape": list(text_features.shape),
                "device": "cpu",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
