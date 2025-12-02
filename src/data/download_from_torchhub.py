# these are the models we will download from torchvision


if __name__ == "__main__":
    import torch
    import torchvision.models as models
    import os

    torchvision_models = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    os.makedirs("../../models/resnet", exist_ok=True)

    for model in torchvision_models:
        models.__dict__[model](pretrained=True)
        print(f"Downloaded {model} from torchvision")
        print("-----   -------------------")
        torch.save(
            models.__dict__[model](pretrained=True),
            f"../../models/resnet/{model}_pretrained.pth",
        )
