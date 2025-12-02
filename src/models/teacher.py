import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 2  # (only melanoma vs not melanoma)

# Define the Teacher model using a pre-trained ResNet50 architecture
class TeacherModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(TeacherModel, self).__init__()
        # Load the pre-trained ResNet50 model
        self.model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
# run a full fine tune on the teacher model
def fine_tune_teacher(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model = TeacherModel()