import argparse
import logging
import pathlib
import sys

import torch
import torchvision.models as tm

ROOT = pathlib.Path(__file__).parent.parent.resolve()
DEFAULT_OUT = ROOT / "models" / "pretrained"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_models(out_dir: pathlib.Path, models_list):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in models_list:
        logger.info("Processing %s", name)
        # determine subfolder based on model name
        if name.startswith("efficientnet"):
            sub = out_dir / "efficient_net"
        elif name.startswith("resnet"):
            sub = out_dir / "resnet"
        elif "mobilenet" in name.lower():
            sub = out_dir / "mobilenet"
        else:
            sub = out_dir / "other"
        sub.mkdir(parents=True, exist_ok=True)
        out_path = sub / f"{name}_pretrained.pth"
        if out_path.exists():
            logger.info("Skipping %s — already downloaded at %s", name, out_path)
            continue

        ctor = getattr(tm, name, None)
        if ctor is None:
            logger.warning("Model %s not found in torchvision.models, skipping.", name)
            continue

        try:
            # Primary attempt (works on many torchvision versions)
            model = ctor(pretrained=True)
            logger.info("Constructed %s with pretrained=True", name)
        except Exception as e:
            logger.warning("pretrained=True failed for %s: %s", name, e)
            try:
                weights_attr = getattr(ctor, "Weights", None)
                if weights_attr is not None:
                    default_weight = getattr(weights_attr, "DEFAULT", None)
                    if default_weight is not None:
                        model = ctor(weights=default_weight)
                        logger.info("Constructed %s with weights=DEFAULT", name)
                    else:
                        raise RuntimeError("No DEFAULT weights available")
                else:
                    model = ctor()
                    logger.info("Constructed %s without pretrained weights (fallback)", name)
            except Exception as e2:
                logger.exception("Failed to construct %s: %s", name, e2)
                continue

        try:
            torch.save(model.state_dict(), str(out_path))
            logger.info("Saved state_dict to %s", out_path)
        except Exception as e:
            logger.exception("Failed to save %s: %s", name, e)


def main(argv=None):
    p = argparse.ArgumentParser(description="Download EfficientNet models from torchvision and save state_dicts.")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output directory")
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "mobilenet_v2",
            "mobilenet_v3_large",
            "mobilenet_v3_small",
        ],
        help="List of torchvision model names to download",
    )
    args = p.parse_args(argv)
    out_dir = pathlib.Path(args.out).expanduser().resolve()
    download_models(out_dir, args.models)


if __name__ == "__main__":
    main()