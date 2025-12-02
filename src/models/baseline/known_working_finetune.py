# ============================================================================
# Part 5: Transfer Learning (20 points)
# ============================================================================

import pathlib

import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models

model_dir = pathlib.Path(__file__).parent.parent / "models"


def make_dataloaders(debug_: bool = False) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders for the UVA landmarks dataset."""

    def download_dataset():
        """Download and extract the UVA landmarks dataset into ../dataset.

        This saves dataset.zip into the repository root (one level above src) and
        extracts it there so the dataset root becomes ../dataset.
        """
        url = "https://firebasestorage.googleapis.com/v0/b/uva-landmark-images.appspot.com/o/dataset.zip?alt=media&token=e1403951-30d6-42b8-ba4e-394af1a2ddb7"

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_dir = os.path.join(repo_root, "dataset")
        zip_path = os.path.join(repo_root, "dataset.zip")

        if not os.path.exists(dataset_dir):
            print("Downloading dataset...")
            urllib.request.urlretrieve(url, zip_path)
            print("Extracting dataset to:", repo_root)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(repo_root)
            try:
                os.remove(zip_path)
            except Exception:
                pass
        else:
            print("Dataset already exists at", dataset_dir)

    # Ensure dataset is present (auto-download if missing)
    repo_dataset_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset")
    )
    if not os.path.exists(repo_dataset_dir):
        try:
            download_dataset()
        except Exception as e:
            if debug_:
                print(f"Warning: failed to download dataset automatically: {e}")

    # Dataset parameters
    print("current working directory:", os.getcwd())
    data_dir = repo_dataset_dir
    batch_size = 32
    img_height = 150
    img_width = 150
    #   num_classes = 18

    # Class names for UVA landmarks
    # class_names = [
    #     "AcademicalVillage",
    #     "AldermanLibrary",
    #     "AlumniHall",
    #     "AquaticFitnessCenter",
    #     "BavaroHall",
    #     "BrooksHall",
    #     "ClarkHall",
    #     "MadisonHall",
    #     "MinorHall",
    #     "NewCabellHall",
    #     "NewcombHall",
    #     "OldCabellHall",
    #     "OlssonHall",
    #     "RiceHall",
    #     "Rotunda",
    #     "ScottStadium",
    #     "ThorntonHall",
    #     "UniversityChapel",
    # ]

    # Data transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    # Find correct root that contains class subdirectories (handles nested zip layouts)
    def _find_class_root(root, max_depth=4):
        exts = {".jpg", ".jpeg"}

        def _contains_images(dirpath):
            try:
                for entry in os.listdir(dirpath):
                    p = os.path.join(dirpath, entry)
                    if os.path.isfile(p) and os.path.splitext(entry)[1].lower() in exts:
                        return True
            except Exception:
                return False
            return False

        current = root
        for _ in range(max_depth):
            # list subdirectories
            try:
                subdirs = [
                    d
                    for d in os.listdir(current)
                    if os.path.isdir(os.path.join(current, d))
                ]
            except Exception:
                break

            # If subdirs themselves contain image files, current is the class root
            if any(_contains_images(os.path.join(current, d)) for d in subdirs):
                return current

            # If only one subdir, descend into it and continue searching
            if len(subdirs) == 1:
                current = os.path.join(current, subdirs[0])
                continue

            # Otherwise, check one more level: if any subdir has subdirs that contain images, use that
            for d in subdirs:
                subpath = os.path.join(current, d)
                try:
                    subsubdirs = [
                        sd
                        for sd in os.listdir(subpath)
                        if os.path.isdir(os.path.join(subpath, sd))
                    ]
                except Exception:
                    continue
                if any(
                    _contains_images(os.path.join(subpath, sd)) for sd in subsubdirs
                ):
                    return subpath

            break

        return current

    class_root = _find_class_root(data_dir)
    if debug_:
        print(f"Using class root for ImageFolder: {class_root}")

    full_dataset = datasets.ImageFolder(class_root)

    if len(full_dataset.classes) <= 1:
        raise ValueError(
            "Warning: ImageFolder found the following classes:", full_dataset.classes
        )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Attach class label mappings to loaders for easy lookup
    classes = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_loader.classes = classes
    train_loader.class_to_idx = class_to_idx
    train_loader.idx_to_class = idx_to_class

    val_loader.classes = classes
    val_loader.class_to_idx = class_to_idx
    val_loader.idx_to_class = idx_to_class

    if debug_:
        # Print mapping for verification
        print("Class to index mapping:")
        for cls, idx in class_to_idx.items():
            print(f"  {cls}: {idx}")

    if debug_:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader



def get_pretrained_model(model_name="resnet18", num_classes=18, feature_extract=True):
    """
    (20 pts): Implement transfer learning with pretrained models.

     Transfer Learning: Use features learned on ImageNet for our task!

     Two strategies:
     1. Feature Extraction (feature_extract=True):
        - Freeze all conv layers (no gradient updates)
        - Only train the new classifier
        - Fast training, good for small datasets
        - Use when: limited data, similar domain

     2. Fine-tuning (feature_extract=False):
        - Initialize with pretrained weights
        - Allow all layers to update
        - Slower training but can achieve higher accuracy
        - Use when: more data, different domain

     Implementation steps for each model type:

     ResNet18:
     1. Load: models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
     2. Freeze: Loop through parameters, set requires_grad=False
     3. Replace: model.fc = nn.Linear(model.fc.in_features, num_classes)

     VGG16:
     1. Load: models.vgg16(weights=models.VGG16_Weights.DEFAULT)
     2. Freeze: Focus on model.features (conv layers)
     3. Replace: model.classifier[6] (last linear layer)

     MobileNetV2:
     1. Load: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
     2. Freeze: All parameters
     3. Replace: model.classifier[1] (last linear layer)

     Tips:
     - Always replace the final layer to match num_classes
     - When fine-tuning, use lower learning rate (1e-4 or 1e-5)
     - Consider unfreezing only last few layers for better results

     Expected accuracy:
     - Feature extraction: 85-90% in 5 epochs
     - Fine-tuning: 90-95% in 10 epochs
    """
    # TODO: Check model_name and load appropriate pretrained model
    # TODO: Implement freezing logic based on feature_extract flag
    # TODO: Replace final classifier layer
    # TODO: Return the modified model
    # Don't forget error handling for unsupported model names!

    # Validate model name
    assert model_name in ["resnet18", "vgg16", "mobilenet_v2"], (
        f"Unsupported model_name: {model_name} - use one of ['resnet18', 'vgg16', 'mobilenet_v2']"
    )

    if model_name == "resnet18":
        print("Loading ResNet18 pretrained model...")
        model = models.resnet18(weights=None)
        state = torch.load(model_dir / "resnet18-pretrained.pth", map_location="cpu")
        model.load_state_dict(state)

        # Freeze parameters only if feature_extract is True
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the classifier (always)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16":
        print("Loading VGG16 pretrained model...")
        model = models.vgg16(weights=None)
        state = torch.load(model_dir / "vgg16-pretrained.pth", map_location="cpu")
        model.load_state_dict(state)

        if feature_extract:
            for param in model.features.parameters():
                param.requires_grad = False

        # Replace the last classifier layer
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == "mobilenet_v2":
        print("Loading MobileNetV2 pretrained model...")
        model = models.mobilenet_v2(weights=None)
        state = torch.load(
            model_dir / "mobilenet_v2-pretrained.pth", map_location="cpu"
        )
        model.load_state_dict(state)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final classifier layer
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


# Test the function here
if __name__ == "__main__":
    # prefer MPS (Apple Silicon) if available, then CUDA, otherwise CPU
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    for model_name in ["resnet18", "vgg16", "mobilenet_v2"]:
        model = get_pretrained_model(model_name, num_classes=18, feature_extract=True)
        print(f"running - {model_name} on device={device}")
        try:
            model = model.to(device)
        except Exception as e:
            print(
                f"Warning: failed to move model to {device}: {e}. Falling back to cpu."
            )
            device = "cpu"
            model = model.to(device)

        try:
            model_summary = summary(
                model,
                input_size=(1, 3, 224, 224),
                device=device,
                col_names=("input_size", "output_size", "num_params", "trainable"),
            )
            print(model_summary)
        except Exception as e:
            print(f"torchinfo.summary failed for {model_name} on device={device}: {e}")
            print(model)

    for model_name in ["resnet18", "vgg16", "mobilenet_v2"]:
        model = get_pretrained_model(model_name, num_classes=18, feature_extract=False)
        print(f"running - {model_name} on device={device}")
        try:
            model = model.to(device)
        except Exception as e:
            print(
                f"Warning: failed to move model to {device}: {e}. Falling back to cpu."
            )
            device = "cpu"
            model = model.to(device)

        try:
            model_summary = summary(
                model,
                input_size=(1, 3, 224, 224),
                device=device,
                col_names=("input_size", "output_size", "num_params", "trainable"),
            )
            print(model_summary)
        except Exception as e:
            print(f"torchinfo.summary failed for {model_name} on device={device}: {e}")
            print(model)
