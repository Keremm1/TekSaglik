import os
import torch
from torchvision import transforms
from utils import get_loader
from utils import evaluate_model
from modules import CNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "stroke_detection_model.pth"
test_dir = "test_dataset"
img_size = (256, 256)

model = CNNModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_loader_head_ct = get_loader(os.path.join(test_dir, "head_ct"), transform=transform, batch_size=32, shuffle=False)
test_loader_brain_ct = get_loader(os.path.join(test_dir, "brain_ct"), transform=transform, batch_size=32, shuffle=False)

accuracy_head_ct = evaluate_model(model, test_loader_head_ct, device)
accuracy_brain_ct = evaluate_model(model, test_loader_brain_ct, device)

total_accuracy = (accuracy_head_ct + accuracy_brain_ct) / 2

print(f'Head CT Test Accuracy: {accuracy_head_ct * 100:.2f}%')
print(f'Brain CT Test Accuracy: {accuracy_brain_ct * 100:.2f}%')
print(f'Total Accuracy: {total_accuracy * 100:.2f}%')
