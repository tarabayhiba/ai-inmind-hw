"""Supervised fine-tuning of a LeJEPA-pretrained encoder on labeled
CIFAR-10. Loads the encoder checkpoint written by pretrain_lejepa.py,
attaches a linear classifier head to its embed() output (model.py's
LeJEPAClassifier), and trains/evaluates
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from lejepa_data import CIFAR10_MEAN, CIFAR10_STD
from model import CIFARResNetEncoder, LeJEPAClassifier

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def get_loaders(finetune_cfg):
    """ Builds Subsets of two separately-transformed CIFAR10 instances over an identical index split, 
    val stays free of train's crop/flip augmentation."""
    train_dir = config['paths']['train_dir']
    test_dir = config['paths']['test_dir']
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    full_train_aug = datasets.CIFAR10(root=train_dir, train=True, download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=train_dir, train=True, download=True, transform=eval_transform)
    dataset_test = datasets.CIFAR10(root=test_dir, train=False, download=True, transform=eval_transform)

    val_split = finetune_cfg.get('val_split', 0.1)
    n_total = len(full_train_aug)
    n_val = int(n_total * val_split)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
    val_indices, train_indices = indices[:n_val], indices[n_val:]

    dataset_train = Subset(full_train_aug, train_indices)
    dataset_val = Subset(full_train_eval, val_indices)

    loader_kwargs = dict(batch_size=finetune_cfg['batch_size'], num_workers=finetune_cfg['num_workers'])
    dataloader_train = DataLoader(dataset_train, shuffle=True, **loader_kwargs)
    dataloader_val = DataLoader(dataset_val, shuffle=False, **loader_kwargs)
    dataloader_test = DataLoader(dataset_test, shuffle=False, **loader_kwargs)
    return dataloader_train, dataloader_val, dataloader_test


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total, 100 * correct / total


def finetune(model, dataloader_train, dataloader_val, criterion, optimizer, device, epochs, checkpoint_path):
    """Trains for `epochs`, saving a checkpoint each time val accuracy
    improves (best-by-val-accuracy, not just the final epoch)"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_acc = -1.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        with tqdm(
            dataloader_train,
            desc=f"Finetune epoch {epoch+1}/{epochs}",
            leave=True,
            unit="batch"
        ) as progress_bar:
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / (i + 1)})

        avg_train_loss = running_loss / len(dataloader_train)
        val_loss, val_acc = evaluate(model, dataloader_val, criterion, device)
        print(f"Epoch {epoch+1} finished. Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | Val acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best val acc {val_acc:.2f}% -- checkpoint saved to {checkpoint_path}")

    print(f'Finished fine-tuning. Best val acc: {best_val_acc:.2f}%')
    return best_val_acc


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    lejepa_cfg = config['lejepa']
    finetune_cfg = lejepa_cfg['finetune']
    checkpoint_path = config['paths']['lejepa_finetuned_path']

    dataloader_train, dataloader_val, dataloader_test = get_loaders(finetune_cfg)

    encoder = CIFARResNetEncoder(
        embedding_dim=lejepa_cfg['embedding_dim'],
        proj_dim=lejepa_cfg['proj_dim'],
    )
    encoder_path = config['paths']['lejepa_encoder_path']
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    print(f"Loaded pretrained encoder from {encoder_path}")

    model = LeJEPAClassifier(encoder, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=finetune_cfg['lr'],
        weight_decay=finetune_cfg['weight_decay'],
    )

    finetune(model, dataloader_train, dataloader_val, criterion, optimizer, device, finetune_cfg['epochs'], checkpoint_path)

    # Reload the best-val-acc checkpoint (not necessarily the last epoch) for the final test number
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc = evaluate(model, dataloader_test, criterion, device)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
