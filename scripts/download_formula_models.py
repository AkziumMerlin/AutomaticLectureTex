#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import yaml


UNIMERNET_VARIANTS = {
    "small": {
        "repo": "wanderkid/unimernet_small",
        "checkpoint": "unimernet_small.pth",
        "sha256": "fa54b0a8126bb60060bc90818ce20a5ca1b5dd5d7da5c0983579f5c3a2cc90ea",
    },
    "base": {
        "repo": "wanderkid/unimernet_base",
        "checkpoint": "pytorch_model.pth",
        "sha256": "16cd0891233cfee3c11215a7b87306f160f7e7f3f52091a6253751c149a8c180",
    },
}

MFD_REPO = "opendatalab/PDF-Extract-Kit-1.0"
MFD_FILE = "models/MFD/YOLO/yolo_v8_ft.pt"
MFD_SHA256 = "41029d5abb9b0b6df825ecaf8adf9151762ea0688ac0fc4ea0ea34ab9a5808fc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _write_unimernet_config(model_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = model_dir / checkpoint_name
    payload = {
        "model": {
            "arch": "unimernet",
            "model_type": "unimernet",
            "model_config": {
                "model_name": str(model_dir.resolve()),
                "max_seq_len": 1536,
            },
            "load_pretrained": True,
            "pretrained": str(checkpoint.resolve()),
            "tokenizer_config": {"path": str(model_dir.resolve())},
        },
        "datasets": {
            "formula_rec_eval": {
                "vis_processor": {
                    "eval": {
                        "name": "formula_image_eval",
                        "image_size": [192, 672],
                    }
                }
            }
        },
        "run": {
            "runner": "runner_iter",
            "task": "unimernet_train",
            "batch_size_train": 1,
            "batch_size_eval": 1,
            "num_workers": 1,
            "iters_per_inner_epoch": 1,
            "max_iters": 1,
            "seed": 42,
            "output_dir": str((model_dir / "_runtime_output").resolve()),
            "evaluate": True,
            "test_splits": ["eval"],
            "device": "cuda",
            "world_size": 1,
            "dist_url": "env://",
            "distributed": False,
            "distributed_type": "ddp",
            "generate_cfg": {"temperature": 0.0},
        },
    }
    config_path = model_dir / "automatic_lecture_tex.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download UniMERNet weights and the PDF-Extract-Kit MFD weights."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/formula"),
        help="Destination root (default: models/formula).",
    )
    parser.add_argument(
        "--unimernet-size",
        choices=sorted(UNIMERNET_VARIANTS),
        default="base",
        help="UniMERNet variant to download. 'base' is the largest official checkpoint.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: "
            "pip install -e '.[formula-vision]'"
        ) from exc

    root = args.output_dir.resolve()
    variant = UNIMERNET_VARIANTS[args.unimernet_size]
    checkpoint_name = str(variant["checkpoint"])
    unimernet_dir = root / f"unimernet_{args.unimernet_size}"
    mfd_dir = root / "mfd"
    unimernet_dir.mkdir(parents=True, exist_ok=True)
    mfd_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=str(variant["repo"]),
        local_dir=unimernet_dir,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            checkpoint_name,
        ],
    )
    checkpoint = unimernet_dir / checkpoint_name
    _verify(checkpoint, str(variant["sha256"]))

    mfd_download = Path(
        hf_hub_download(
            repo_id=MFD_REPO,
            filename=MFD_FILE,
        )
    )
    mfd_target = mfd_dir / "yolo_v8_ft.pt"
    if mfd_download.resolve() != mfd_target.resolve():
        shutil.copy2(mfd_download, mfd_target)
    _verify(mfd_target, MFD_SHA256)

    config_path = _write_unimernet_config(unimernet_dir, checkpoint_name)

    print(f"UniMERNet-{args.unimernet_size} model: {unimernet_dir}")
    print(f"UniMERNet config: {config_path}")
    print(f"MFD weights: {mfd_target}")
    print()
    print("Suggested config:")
    print("vision:")
    print("  formula_detection:")
    print("    enabled: true")
    print("    backend: yolov8")
    print(f"    model_path: {mfd_target}")
    print("  math_ocr:")
    print("    backend: unimernet")
    print(f"    unimernet_config_path: {config_path}")


if __name__ == "__main__":
    main()
