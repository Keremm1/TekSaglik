from modules import CoAtNet, CNNModel
import torch
import torch.nn as nn
from lion_pytorch import Lion
import torch.optim as optim
import torchvision.transforms as transforms
from utils import get_loader
from utils import train_model, plot_metrics

data_dir = "dataset"
img_size = (224, 224)
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

model = CNNModel(img_size=img_size).to(device) if input("Enter the model you want to use (CoAtNet/CNN): ") == "CNN" else CoAtNet(image_size=img_size, num_classes=1).to(device)
optimizer_class = Lion if input("Enter the optimizer you want to use (Lion/Adam): ") == "Lion" else optim.Adam

model_path = f"stroke_detection_model_{model.__class__.__name__}.pth"

criterion = nn.BCELoss()
learning_rate = 1e-3
num_epochs = 10

train_loader, val_loader = get_loader(data_dir, transform=transform, split_ratio=0.8, batch_size=batch_size, shuffle=True)

train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs, 
                                                                         criterion, optimizer_class, learning_rate, 
                                                                         model_path=model_path, device=device)

plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)