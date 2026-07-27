from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"

sys.path.insert(0, str(CORE_ROOT))

from config import cfg as default_cfg, update_config
from nets import build_spnv2
from utils.utils import set_seeds_cudnn


TRANSFER_PREFIXES = (
    "backbone.efficientnet.",
    "backbone.bifpn.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(loaded, dict) and "state_dict" in loaded:
        state = loaded["state_dict"]
    else:
        state = loaded

    if not isinstance(state, dict):
        raise TypeError(
            f"{path} does not contain a valid state_dict."
        )

    if state and all(key.startswith("module.") for key in state):
        state = {
            key[len("module."):]: value
            for key, value in state.items()
        }

    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a full Tango SPNv2 initialization checkpoint "
            "using SPE3R EfficientNet and BiFPN weights."
        )
    )

    parser.add_argument(
        "--tango-cfg",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--spe3r-checkpoint",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    tango_cfg_path = args.tango_cfg.resolve()
    spe3r_path = args.spe3r_checkpoint.resolve()
    output_path = args.output.resolve()
    manifest_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )

    if not tango_cfg_path.exists():
        raise FileNotFoundError(
            f"Tango configuration not found: {tango_cfg_path}"
        )

    if not spe3r_path.exists():
        raise FileNotFoundError(
            f"SPE3R checkpoint not found: {spe3r_path}"
        )

    if output_path.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing initialization file: "
            f"{output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build fresh Tango model deterministically
    # --------------------------------------------------------

    tango_cfg = default_cfg.clone()

    config_args = SimpleNamespace(
        cfg=str(tango_cfg_path),
        opts=[],
    )

    update_config(tango_cfg, config_args)

    tango_cfg.defrost()
    tango_cfg.MODEL.PRETRAIN_FILE = ""
    tango_cfg.AUTO_RESUME = False
    tango_cfg.freeze()

    seed = int(tango_cfg.SEED)

    set_seeds_cudnn(
        tango_cfg,
        seed=seed,
    )

    print("Building fresh Tango SPNv2 model...")
    print("Seed:", seed)

    tango_model = build_spnv2(tango_cfg)
    target_state = tango_model.state_dict()

    fresh_heads = {
        key: value.detach().clone()
        for key, value in target_state.items()
        if key.startswith("heads.")
    }

    # --------------------------------------------------------
    # Load and select SPE3R shared representation
    # --------------------------------------------------------

    source_state = load_state_dict(spe3r_path)

    source_shared = {
        key: value
        for key, value in source_state.items()
        if key.startswith(TRANSFER_PREFIXES)
    }

    target_shared = {
        key: value
        for key, value in target_state.items()
        if key.startswith(TRANSFER_PREFIXES)
    }

    missing_in_tango = sorted(
        set(source_shared) - set(target_shared)
    )

    missing_in_spe3r = sorted(
        set(target_shared) - set(source_shared)
    )

    shape_mismatches = []
    dtype_mismatches = []

    for key in sorted(
        set(source_shared).intersection(target_shared)
    ):
        source_tensor = source_shared[key]
        target_tensor = target_shared[key]

        if source_tensor.shape != target_tensor.shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "spe3r_shape": list(source_tensor.shape),
                    "tango_shape": list(target_tensor.shape),
                }
            )

        if source_tensor.dtype != target_tensor.dtype:
            dtype_mismatches.append(
                {
                    "key": key,
                    "spe3r_dtype": str(source_tensor.dtype),
                    "tango_dtype": str(target_tensor.dtype),
                }
            )

    assert source_shared, (
        "No EfficientNet or BiFPN tensors were selected."
    )

    assert not missing_in_tango, (
        f"{len(missing_in_tango)} selected tensors "
        "are missing in Tango."
    )

    assert not missing_in_spe3r, (
        f"{len(missing_in_spe3r)} Tango shared tensors "
        "are missing in SPE3R."
    )

    assert not shape_mismatches, (
        f"{len(shape_mismatches)} shape mismatches detected."
    )

    assert not dtype_mismatches, (
        f"{len(dtype_mismatches)} dtype mismatches detected."
    )

    assert not any(
        key.startswith("heads.")
        for key in source_shared
    ), "A head tensor was accidentally selected."

    # --------------------------------------------------------
    # Create complete Tango initialization
    # --------------------------------------------------------

    merged_state = dict(target_state)
    merged_state.update(source_shared)

    tango_model.load_state_dict(
        merged_state,
        strict=True,
    )

    final_state = tango_model.state_dict()

    # Verify all selected SPE3R tensors were copied exactly.
    for key, source_tensor in source_shared.items():
        assert torch.equal(
            final_state[key],
            source_tensor,
        ), f"Transferred tensor differs: {key}"

    # Verify all Tango-specific heads remained untouched.
    for key, fresh_tensor in fresh_heads.items():
        assert torch.equal(
            final_state[key],
            fresh_tensor,
        ), f"Tango head changed unexpectedly: {key}"

    # --------------------------------------------------------
    # Save atomically
    # --------------------------------------------------------

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".temporary"
    )

    torch.save(
        final_state,
        temporary_path,
    )

    # Reload before accepting the file.
    saved_state = load_state_dict(temporary_path)

    assert set(saved_state) == set(final_state), (
        "Saved initialization has different state_dict keys."
    )

    for key in final_state:
        assert torch.equal(
            saved_state[key],
            final_state[key],
        ), f"Saved tensor verification failed: {key}"

    temporary_path.replace(output_path)

    shared_elements = sum(
        tensor.numel()
        for tensor in source_shared.values()
    )

    head_elements = sum(
        tensor.numel()
        for tensor in fresh_heads.values()
    )

    manifest = {
        "purpose": (
            "Tango SPNv2 initialization using SPE3R-trained "
            "EfficientNet and BiFPN with fresh Tango heads."
        ),
        "tango_config": str(tango_cfg_path),
        "spe3r_checkpoint": str(spe3r_path),
        "spe3r_checkpoint_sha256": sha256_file(spe3r_path),
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": sha256_file(output_path),
        "seed": seed,
        "transfer_prefixes": list(TRANSFER_PREFIXES),
        "transferred_tensor_count": len(source_shared),
        "transferred_element_count": shared_elements,
        "fresh_head_tensor_count": len(fresh_heads),
        "fresh_head_element_count": head_elements,
        "missing_in_tango": missing_in_tango,
        "missing_in_spe3r": missing_in_spe3r,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "strict_full_model_load_passed": True,
        "saved_file_reload_passed": True,
        "tango_heads_unchanged": True,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nTRANSFER INITIALIZATION CREATED")
    print("-------------------------------")
    print("Output:", output_path)
    print("Manifest:", manifest_path)
    print("Transferred tensors:", len(source_shared))
    print(f"Transferred elements: {shared_elements:,}")
    print("Fresh Tango head tensors:", len(fresh_heads))
    print(f"Fresh Tango head elements: {head_elements:,}")
    print(
        "Output SHA256:",
        manifest["output_checkpoint_sha256"],
    )
    print("\nPERMANENT TRANSFER VERIFICATION PASSED")


if __name__ == "__main__":
    main()
