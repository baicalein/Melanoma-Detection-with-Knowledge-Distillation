# this needs an nvidia gpu
import os
from pathlib import Path
import torch

if __name__ == "__main__":

    model_type = "GPUNet-0"  # select one
    precision = "fp32"       # "fp32" or "fp16"

    # ensure torch cache is inside the project so cached files are easy to find/wget
    project_root = Path(__file__).resolve().parents[2]
    project_cache = project_root / ".cache" / "torch"
    os.environ.setdefault("TORCH_HOME", str(project_cache))
    project_cache.mkdir(parents=True, exist_ok=True)

    # download model (will cache weights under TORCH_HOME)
    gpunet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_gpunet",
        pretrained=True,
        model_type=model_type,
        model_math=precision,
    )
    gpunet.eval()

    # save a local copy of the state dict
    out_dir = project_root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"gpunet_{model_type}_{precision}.pt"
    torch.save(gpunet.state_dict(), str(out_file))
    print(f"Saved model state_dict to {out_file}")

    # list cache files (you can wget these from the filesystem or from an HTTP URL if available)
    hub_dir = Path(torch.hub.get_dir())
    print("Torch hub cache directory:", hub_dir)
    for p in sorted(set(hub_dir.rglob("*"))):
        if p.is_file() and ("gpunet" in p.name or p.suffix in {".pt", ".pth", ".ckpt"}):
            print(p)

    
    ## Vit
# https://huggingface.co/google/vit-base-patch16-224

# git clone https://huggingface.co/google/vit-base-patch16-224

# ## derm foundation
# https://huggingface.co/google/derm-foundation

# # git-xet install
# brew tap huggingface/tap
# brew install git-xet
# git xet install