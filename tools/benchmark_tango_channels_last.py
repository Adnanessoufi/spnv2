from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path("core").resolve()))

from config import cfg as default_cfg, update_config
from dataset import get_dataloader
from nets import build_spnv2
from solver import get_optimizer
from utils.utils import set_seeds_cudnn


CONFIG_PATH = Path(
    "experiments/"
    "offline_train_full_config_phi3_BN_spe3r_windows.yaml"
).resolve()

WARMUP_STEPS = 5
TIMED_STEPS = 15
MAX_ATTEMPTS = 40
NUM_BATCHES_PER_EPOCH = 11991


def load_config():
    cfg = default_cfg.clone()

    update_config(
        cfg,
        SimpleNamespace(
            cfg=str(CONFIG_PATH),
            opts=[],
        ),
    )

    cfg.defrost()
    cfg.TRAIN.WORKERS = 0
    cfg.AUTO_RESUME = False
    cfg.VERBOSE = False
    cfg.freeze()

    return cfg


def run_mode(
    cfg,
    images,
    targets,
    use_channels_last,
):
    mode_name = (
        "CHANNELS_LAST"
        if use_channels_last
        else "STANDARD_NCHW"
    )

    print(f"\n{'=' * 60}")
    print(mode_name)
    print("=" * 60)

    set_seeds_cudnn(
        cfg,
        seed=cfg.SEED,
    )

    device = torch.device("cuda", 0)

    model = build_spnv2(cfg).to(device)

    if use_channels_last:
        model = model.to(
            memory_format=torch.channels_last
        )

    model.train()

    optimizer = get_optimizer(
        cfg,
        model,
    )

    # The previous tests showed that 32 is the stable scale.
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=32.0,
        growth_interval=2000,
    )

    if use_channels_last:
        test_images = images.contiguous(
            memory_format=torch.channels_last
        )
    else:
        test_images = images.contiguous()

    first_conv = next(
        module
        for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    )

    observed_format = {
        "channels_last": None,
    }

    def check_format(module, inputs):
        tensor = inputs[0]

        observed_format["channels_last"] = (
            tensor.is_contiguous(
                memory_format=torch.channels_last
            )
        )

    hook = first_conv.register_forward_pre_hook(
        check_format
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    successful_warmups = 0
    successful_timings = 0
    attempts = 0
    timings = []
    first_loss = None

    while (
        successful_timings < TIMED_STEPS
        and attempts < MAX_ATTEMPTS
    ):
        attempts += 1

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.amp.autocast(
            "cuda",
            enabled=True,
        ):
            loss, loss_items = model(
                test_images,
                is_train=True,
                gpu=device,
                **targets,
            )

        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"{mode_name}: non-finite forward loss"
            )

        for name, value in loss_items.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"{mode_name}: non-finite {name} loss"
                )

        scale_before = scaler.get_scale()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        scale_after = scaler.get_scale()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if scale_after < scale_before:
            print(
                f"Handled overflow: "
                f"{scale_before:.0f} -> "
                f"{scale_after:.0f}"
            )
            continue

        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"{mode_name}: non-finite gradient"
            )

        if first_loss is None:
            first_loss = float(
                loss.detach().cpu()
            )

        if successful_warmups < WARMUP_STEPS:
            successful_warmups += 1

            print(
                f"Warm-up "
                f"{successful_warmups}/{WARMUP_STEPS}: "
                f"{elapsed:.3f} s"
            )

            continue

        successful_timings += 1
        timings.append(elapsed)

        print(
            f"Timed step "
            f"{successful_timings}/{TIMED_STEPS}: "
            f"{elapsed:.3f} s"
        )

    hook.remove()

    if successful_timings < TIMED_STEPS:
        raise RuntimeError(
            f"{mode_name}: only "
            f"{successful_timings} successful timed steps"
        )

    median_time = statistics.median(timings)
    mean_time = statistics.mean(timings)

    epoch_hours = (
        median_time
        * NUM_BATCHES_PER_EPOCH
        / 3600
    )

    peak_memory_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(f"\n{mode_name} RESULT")
    print("First loss:", first_loss)
    print(
        "First convolution received channels-last:",
        observed_format["channels_last"],
    )
    print(
        f"Median step: {median_time:.3f} s"
    )
    print(
        f"Mean step:   {mean_time:.3f} s"
    )
    print(
        f"Estimated epoch: {epoch_hours:.2f} hours"
    )
    print(
        f"Peak allocated VRAM: "
        f"{peak_memory_gb:.2f} GB"
    )

    result = {
        "name": mode_name,
        "median": median_time,
        "mean": mean_time,
        "epoch_hours": epoch_hours,
        "loss": first_loss,
        "channels_last_observed": (
            observed_format["channels_last"]
        ),
    }

    del optimizer
    del scaler
    del model

    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():
    assert CONFIG_PATH.exists()
    assert torch.cuda.is_available()

    torch.cuda.set_device(0)

    cfg = load_config()

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print("Loading one real Tango batch...")

    loader = get_dataloader(
        cfg,
        split="train",
        distributed=False,
        load_labels=True,
    )

    # Use stylized images for this benchmark.
    loader.dataset.styleAug = True
    loader.dataset.style_prob = 1.0

    images, targets = next(iter(loader))

    assert images.shape[0] == 4

    print("Batch shape:", tuple(images.shape))

    standard = run_mode(
        cfg,
        images,
        targets,
        use_channels_last=False,
    )

    channels_last = run_mode(
        cfg,
        images,
        targets,
        use_channels_last=True,
    )

    improvement = (
        standard["median"]
        - channels_last["median"]
    ) / standard["median"] * 100

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        f"Standard median:      "
        f"{standard['median']:.3f} s"
    )

    print(
        f"Channels-last median: "
        f"{channels_last['median']:.3f} s"
    )

    print(
        f"Speed improvement:    "
        f"{improvement:.1f}%"
    )

    print(
        f"Standard epoch:       "
        f"{standard['epoch_hours']:.2f} hours"
    )

    print(
        f"Channels-last epoch:  "
        f"{channels_last['epoch_hours']:.2f} hours"
    )

    if not channels_last[
        "channels_last_observed"
    ]:
        raise RuntimeError(
            "Channels-last did not reach the first convolution."
        )

    relative_loss_difference = abs(
        standard["loss"]
        - channels_last["loss"]
    ) / max(abs(standard["loss"]), 1e-12)

    print(
        f"Initial-loss difference: "
        f"{relative_loss_difference * 100:.3f}%"
    )

    if improvement >= 15:
        print(
            "\nDECISION: Channels-last is promising."
        )
    elif improvement >= 5:
        print(
            "\nDECISION: Small improvement only."
        )
    else:
        print(
            "\nDECISION: Channels-last is not useful here."
        )

    print("\nNo checkpoint or training output was created.")


if __name__ == "__main__":
    main()
