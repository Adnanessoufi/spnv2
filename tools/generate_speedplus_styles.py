from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


EXPECTED_SIZE = (768, 512)
CHECKPOINT_NAMES = (
    "checkpoint_transformer.pth",
    "checkpoint_stylepredictor.pth",
    "checkpoint_embeddings.pth",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def filename_seed(base_seed: int, name: str) -> int:
    value = hashlib.sha256(
        f"{base_seed}:{name}".encode("utf-8")
    ).digest()

    return int.from_bytes(
        value[:8],
        byteorder="little",
        signed=False,
    ) % (2**63 - 1)


def read_names(csv_path: Path) -> list[str]:
    names = []

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.reader(handle):
            if row:
                names.append(row[0].strip())

    if not names:
        raise RuntimeError("Training CSV contains no images.")

    if len(names) != len(set(names)):
        raise RuntimeError(
            "Duplicate filenames detected in training CSV."
        )

    return names


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")

        if image.size != EXPECTED_SIZE:
            raise RuntimeError(
                f"Wrong source size for {path}: {image.size}"
            )

        array = np.array(
            image,
            dtype=np.uint8,
            copy=True,
        )

    return (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        .div_(255.0)
    )


def save_jpeg(
    tensor: torch.Tensor,
    output_path: Path,
) -> None:
    array = (
        tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )

    temporary_path = output_path.with_name(
        output_path.name + ".temporary"
    )

    Image.fromarray(
        array,
        mode="RGB",
    ).save(
        temporary_path,
        format="JPEG",
        quality=95,
    )

    with Image.open(temporary_path) as check:
        check.load()

        if check.mode != "RGB":
            raise RuntimeError(
                f"Saved image is not RGB: {temporary_path}"
            )

        if check.size != EXPECTED_SIZE:
            raise RuntimeError(
                f"Wrong saved size: {temporary_path}"
            )

    temporary_path.replace(output_path)


def verify_dataset(
    names: list[str],
    source_dir: Path,
    output_dir: Path,
) -> None:
    expected = set(names)

    actual_paths = list(output_dir.glob("*.jpg"))
    actual = {path.name for path in actual_paths}

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing:
        raise RuntimeError(
            f"{len(missing)} outputs missing. "
            f"First: {missing[:5]}"
        )

    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} unexpected JPG files. "
            f"First: {unexpected[:5]}"
        )

    for index, name in enumerate(names, start=1):
        path = output_dir / name

        if path.stat().st_size == 0:
            raise RuntimeError(f"Empty output: {path}")

        with Image.open(path) as image:
            if image.size != EXPECTED_SIZE:
                raise RuntimeError(
                    f"Wrong output dimensions: {path}"
                )

            image.verify()

        if index % 5000 == 0:
            print(
                f"Verified {index:,}/{len(names):,}"
            )

    # Confirm selected outputs are not copies of originals.
    sample_count = min(20, len(names))

    for name in names[:sample_count]:
        with Image.open(source_dir / name) as original:
            original_array = np.asarray(
                original.convert("RGB"),
                dtype=np.float32,
            )

        with Image.open(output_dir / name) as styled:
            styled_array = np.asarray(
                styled.convert("RGB"),
                dtype=np.float32,
            )

        difference = np.abs(
            styled_array - original_array
        ).mean()

        if difference <= 0.0:
            raise RuntimeError(
                f"Styled image equals original: {name}"
            )

    print("\nFULL STYLE DATASET VERIFICATION PASSED")
    print("Expected images:", len(names))
    print("Generated images:", len(actual))
    print("Missing images: 0")
    print("Unexpected JPG files: 0")
    print("Sample copy checks: PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--style-repo", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--expected-count", type=int, default=47966)

    args = parser.parse_args()

    for path in (
        args.source,
        args.csv,
        args.style_repo,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    names = read_names(args.csv)

    if len(names) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} training images, "
            f"but CSV contains {len(names)}."
        )

    missing_sources = [
        name
        for name in names
        if not (args.source / name).exists()
    ]

    if missing_sources:
        raise RuntimeError(
            f"{len(missing_sources)} source images missing. "
            f"First: {missing_sources[:5]}"
        )

    commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.style_repo),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    expected_commit = (
        "0273e2e66a894d553c7466773802e09eae41f698"
    )

    if commit != expected_commit:
        raise RuntimeError(
            f"Wrong StyleAugmentor commit: {commit}"
        )

    checkpoint_dir = (
        args.style_repo
        / "styleaug"
        / "checkpoints"
    )

    for name in CHECKPOINT_NAMES:
        path = checkpoint_dir / name

        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Missing or empty checkpoint: {path}"
            )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove abandoned temporary files only.
    for temporary in args.output.glob("*.temporary"):
        temporary.unlink()

    existing = {
        path.name
        for path in args.output.glob("*.jpg")
    }

    unexpected_existing = existing - set(names)

    if unexpected_existing:
        raise RuntimeError(
            "Unexpected files already exist in output folder: "
            f"{sorted(unexpected_existing)[:5]}"
        )

    for name in sorted(existing):
        with Image.open(args.output / name) as image:
            image.verify()

    pending = [
        name
        for name in names
        if name not in existing
    ]

    print("Training filenames:", len(names))
    print("Already completed:", len(existing))
    print("Remaining:", len(pending))
    print("Alpha:", args.alpha)
    print("Seed:", args.seed)

    if not pending:
        verify_dataset(
            names,
            args.source,
            args.output,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for full style generation."
        )

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sys.path.insert(
        0,
        str(args.style_repo.resolve()),
    )

    # Compatibility with recent PyTorch torch.load defaults.
    original_torch_load = torch.load

    def compatible_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(
            *load_args,
            **load_kwargs,
        )

    torch.load = compatible_torch_load

    try:
        from styleaug import StyleAugmentor
        augmentor = StyleAugmentor()
    finally:
        torch.load = original_torch_load

    augmentor.eval()

    current_batch_size = args.batch_size
    completed = len(existing)
    position = 0
    start_time = time.time()

    while position < len(pending):
        batch_names = pending[
            position:position + current_batch_size
        ]

        try:
            image_batch = torch.stack(
                [
                    load_rgb(args.source / name)
                    for name in batch_names
                ],
                dim=0,
            ).to(
                device,
                non_blocking=True,
            )

            latent_vectors = []

            for name in batch_names:
                generator = torch.Generator(
                    device="cpu"
                )

                generator.manual_seed(
                    filename_seed(
                        args.seed,
                        name,
                    )
                )

                latent_vectors.append(
                    torch.randn(
                        100,
                        generator=generator,
                        dtype=torch.float32,
                    )
                )

            latent_batch = torch.stack(
                latent_vectors,
                dim=0,
            ).to(device)

            embedding = (
                torch.mm(
                    latent_batch,
                    augmentor.A.transpose(1, 0),
                )
                + augmentor.mean
            )

            with torch.inference_mode():
                styled_batch = augmentor(
                    image_batch,
                    alpha=args.alpha,
                    downsamples=0,
                    embedding=embedding,
                    useStylePredictor=True,
                )

            if not torch.isfinite(styled_batch).all():
                raise RuntimeError(
                    "StyleAugmentor produced NaN or Inf."
                )

        except torch.cuda.OutOfMemoryError:
            del image_batch
            torch.cuda.empty_cache()

            if current_batch_size == 1:
                raise

            current_batch_size = 1

            print(
                "GPU memory limit reached. "
                "Automatically switching to batch size 1."
            )

            continue

        for name, tensor in zip(
            batch_names,
            styled_batch,
        ):
            save_jpeg(
                tensor,
                args.output / name,
            )

        position += len(batch_names)
        completed += len(batch_names)

        del image_batch
        del styled_batch
        del embedding
        del latent_batch

        if completed % 100 == 0 or position == len(pending):
            elapsed = time.time() - start_time
            generated_now = position
            rate = generated_now / max(elapsed, 1e-6)

            remaining = len(pending) - position
            eta_seconds = remaining / max(rate, 1e-6)

            print(
                f"Generated {completed:,}/{len(names):,} | "
                f"{rate:.2f} images/s | "
                f"ETA {eta_seconds / 3600:.2f} hours"
            )

    verify_dataset(
        names,
        args.source,
        args.output,
    )

    manifest = {
        "purpose": (
            "SPEED+ Tango pre-generated style augmentation "
            "for SPNv2 training."
        ),
        "style_repository_commit": commit,
        "alpha": args.alpha,
        "seed": args.seed,
        "image_count": len(names),
        "image_size": list(EXPECTED_SIZE),
        "source_directory": str(args.source.resolve()),
        "output_directory": str(args.output.resolve()),
        "training_csv": str(args.csv.resolve()),
        "filename_list_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
        "checkpoint_sha256": {
            name: sha256_file(checkpoint_dir / name)
            for name in CHECKPOINT_NAMES
        },
        "jpeg_quality": 95,
        "verification_passed": True,
    }

    manifest_path = (
        args.output
        / "_style_generation_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Manifest:", manifest_path)
    print("\nFULL STYLE GENERATION COMPLETED")


if __name__ == "__main__":
    main()
