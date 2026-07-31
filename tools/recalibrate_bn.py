import argparse
import os
import os.path as osp

import torch
import torch.nn as nn

import _init_paths

from config import cfg, update_config
from nets import build_spnv2
from dataset import get_dataloader
from utils.utils import set_seeds_cudnn


def parse_args():
    parser = argparse.ArgumentParser(description="Recalibrate backbone BatchNorm statistics")
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser.parse_args()


def main():
    args = parse_args()
    update_config(cfg, args)

    set_seeds_cudnn(cfg, seed=cfg.SEED)

    device = torch.device(
        "cuda:0" if cfg.CUDA and torch.cuda.is_available() else "cpu"
    )

    print("Creating SPNv2...")
    model = build_spnv2(cfg)

    print("Loading checkpoint:")
    print(args.checkpoint)

    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    # Everything stays in evaluation mode.
    model.eval()

    # Recalibrate ONLY backbone BatchNorm statistics.
    bn_layers = []

    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()

            # Cumulative moving average over recalibration batches.
            module.momentum = None

            # Only BatchNorm layers enter training mode.
            module.train()

            bn_layers.append(module)

    print(f"Backbone BatchNorm layers found: {len(bn_layers)}")

    if len(bn_layers) == 0:
        raise RuntimeError("No backbone BatchNorm layers found.")

    print("Creating synthetic Tango training dataloader...")

    loader = get_dataloader(
        cfg,
        split="train",
        distributed=False,
        load_labels=True
    )

    print(f"Recalibration batches: {len(loader)}")
    print(f"Batch size: {cfg.TRAIN.IMAGES_PER_GPU}")
    print("Starting BN recalibration...")

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):

            images = images.to(
                device,
                non_blocking=True
            )

            # We only need the shared backbone.
            _ = model.backbone(images)

            if (idx + 1) % 1000 == 0:
                print(
                    f"BN recalibration: "
                    f"{idx + 1}/{len(loader)}"
                )

    print("BN recalibration completed.")

    output_dir = osp.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        args.output
    )

    print("Saved recalibrated checkpoint:")
    print(args.output)


if __name__ == "__main__":
    main()
