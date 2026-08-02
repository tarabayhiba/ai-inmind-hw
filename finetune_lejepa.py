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

from lejepa_data import CIFAR10_MEAN, CIFAR10_STD, get_train_val_indices
from model import CIFARResNetEncoder, LeJEPAClassifier
from pretrain_lejepa import build_scheduler

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


def get_loaders(lejepa_cfg, finetune_cfg):
    """ Builds Subsets of two separately-transformed CIFAR10 instances over an identical index split,
    val stays free of train's crop/flip augmentation. Uses the same shared
    split as pretrain (lejepa_data.get_train_val_indices) so val_indices here
    are exactly the images pretraining excluded -- a genuinely unseen holdout."""
    train_dir = config['paths']['train_dir']
    test_dir = config['paths']['test_dir']
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    full_train_aug = datasets.CIFAR10(root=train_dir, train=True, download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=train_dir, train=True, download=True, transform=eval_transform)
    dataset_test = datasets.CIFAR10(root=test_dir, train=False, download=True, transform=eval_transform)

    train_indices, val_indices = get_train_val_indices(len(full_train_aug), lejepa_cfg['val_split'])

    dataset_train = Subset(full_train_aug, train_indices)
    dataset_val = Subset(full_train_eval, val_indices)

    num_workers = finetune_cfg['num_workers']
    loader_kwargs = dict(
        batch_size=finetune_cfg['batch_size'],
        num_workers=num_workers,
        persistent_workers=num_workers > 0,  # avoid respawning all workers every epoch
    )
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


def save_resume_checkpoint(resume_path, epoch, model, optimizer, scheduler, best_val_acc):
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
    }, resume_path)


def finetune(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device, epochs,
              checkpoint_path, resume_path, start_epoch=0, best_val_acc=-1.0):
    """Trains for `epochs`, saving a checkpoint each time val accuracy
    improves (best-by-val-accuracy, not just the final epoch), plus a
    mid-run resume checkpoint (model+optimizer+scheduler+epoch+best_val_acc)
    after every epoch so a killed/hung run can pick back up."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    for epoch in range(start_epoch, epochs):
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
                scheduler.step()
                running_loss += loss.item()
                progress_bar.set_postfix({'loss': running_loss / (i + 1)})

        avg_train_loss = running_loss / len(dataloader_train)
        val_loss, val_acc = evaluate(model, dataloader_val, criterion, device)
        print(
            f"Epoch {epoch+1} finished. Train loss: {avg_train_loss:.3f} | Val loss: {val_loss:.3f} | "
            f"Val acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best val acc {val_acc:.2f}% -- checkpoint saved to {checkpoint_path}")

        save_resume_checkpoint(resume_path, epoch, model, optimizer, scheduler, best_val_acc)
        print(f"Resume checkpoint saved to {resume_path} (epoch {epoch+1})")

    print(f'Finished fine-tuning. Best val acc: {best_val_acc:.2f}%')
    return best_val_acc


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    lejepa_cfg = config['lejepa']
    finetune_cfg = lejepa_cfg['finetune']
    checkpoint_path = config['paths']['lejepa_finetuned_path']

    dataloader_train, dataloader_val, dataloader_test = get_loaders(lejepa_cfg, finetune_cfg)

    encoder = CIFARResNetEncoder(
        embedding_dim=lejepa_cfg['embedding_dim'],
        proj_dim=lejepa_cfg['proj_dim'],
    )
    encoder_path = config['paths']['lejepa_encoder_path']
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    print(f"Loaded pretrained encoder from {encoder_path}")

    model = LeJEPAClassifier(encoder, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=finetune_cfg['label_smoothing'])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=finetune_cfg['lr'],
        weight_decay=finetune_cfg['weight_decay'],
    )
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=len(dataloader_train),
        epochs=finetune_cfg['epochs'],
        warmup_epochs=finetune_cfg['warmup_epochs'],
    )

    resume_path = config['paths']['lejepa_finetune_resume_path']
    start_epoch = 0
    best_val_acc = -1.0
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from {resume_path} at epoch {start_epoch + 1} (best val acc so far: {best_val_acc:.2f}%)")

    finetune(model, dataloader_train, dataloader_val, criterion, optimizer, scheduler, device, finetune_cfg['epochs'],
             checkpoint_path, resume_path, start_epoch, best_val_acc)

    # Reload the best-val-acc checkpoint (not necessarily the last epoch) for the final test number
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc = evaluate(model, dataloader_test, criterion, device)
    print(f'Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
