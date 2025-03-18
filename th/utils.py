import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer_class, lr=0.001, model_path=None, device='cpu'):
    optimizer = optimizer_class(model.parameters(), lr=lr)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / (i + 1))
        train_accuracies.append(correct / total)

        val_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Training Loss: {train_losses[-1]:.4f}, '
              f'Training Accuracy: {train_accuracies[-1] * 100:.2f}%, '
              f'Validation Accuracy: {val_accuracy * 100:.2f}%')

    if model_path:
        torch.save(model.state_dict(), model_path)

    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def get_loader(data_dir, transform, split_ratio=None, **kwargs):
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    if split_ratio:
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, **kwargs)
        kwargs['shuffle'] = False
        val_loader = DataLoader(val_dataset, **kwargs)
        return train_loader, val_loader
    
    return DataLoader(dataset, **kwargs)

def plot_metrics(losses, accuracies, val_losses=None, val_accuracies=None):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy')
    if val_accuracies:
        plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()