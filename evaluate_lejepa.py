"""Standalone test-set evaluation for a fine-tuned LeJEPA classifier
checkpoint, without re-running fine-tuning."""
import time

import torch
import torch.nn as nn
import yaml

from finetune_lejepa import get_loaders
from model import CIFARResNetEncoder, LeJEPAClassifier

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def evaluate_with_confusion(model, dataloader, criterion, device, num_classes):
    """Like finetune_lejepa.evaluate, but also accumulates a
    (true, predicted) confusion matrix in the same pass."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for true_label, pred_label in zip(labels.view(-1).cpu(), predicted.view(-1).cpu()):
                confusion[true_label, pred_label] += 1
    return total_loss / total, 100 * correct / total, confusion


def print_confusion_matrix(confusion, class_names):
    n = len(class_names)
    codes = [name[:3].upper() for name in class_names]
    label_width = 12
    col_width = 5
    data_width = col_width * n

    top = '┌' + '─' * label_width + '┬' + '─' * data_width + '┐'
    sep = '├' + '─' * label_width + '┼' + '─' * data_width + '┤'
    bottom = '└' + '─' * label_width + '┴' + '─' * data_width + '┘'

    print(top)
    header = ''.join(f'{code:>{col_width}}' for code in codes)
    print(f'│{" " * label_width}│{header}│')
    print(sep)
    for i in range(n):
        row = ''.join(f'{confusion[i, j].item():>{col_width}}' for j in range(n))
        print(f'│{class_names[i]:<{label_width}}│{row}│')
    print(bottom)

    print()
    print('  Legend: ' + ', '.join(f'{code}={name}' for code, name in zip(codes, class_names)))
    print('  Rows = true label, columns = predicted label, diagonal = correct.')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lejepa_cfg = config['lejepa']
    finetune_cfg = lejepa_cfg['finetune']
    checkpoint_path = config['paths']['lejepa_finetuned_path']

    _, _, dataloader_test = get_loaders(lejepa_cfg, finetune_cfg)
    test_dataset = dataloader_test.dataset

    encoder = CIFARResNetEncoder(
        embedding_dim=lejepa_cfg['embedding_dim'],
        proj_dim=lejepa_cfg['proj_dim'],
    )
    model = LeJEPAClassifier(encoder, num_classes=10).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print('=' * 50)
    print('LeJEPA evaluation')
    print('=' * 50)
    print(f'  Checkpoint : {checkpoint_path}')
    print(f'  Device     : {device}')
    print(f'  Dataset    : CIFAR-10 test split ({len(test_dataset)} images, '
          f'{len(test_dataset.classes)} classes, fixed order, never seen in training)')
    print(f'  Batch size : {finetune_cfg["batch_size"]}')
    print('-' * 50)

    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()
    test_loss, test_acc, confusion = evaluate_with_confusion(
        model, dataloader_test, criterion, device, num_classes=len(test_dataset.classes)
    )
    elapsed = time.perf_counter() - start

    print(f'  Test loss  : {test_loss:.3f}')
    print(f'  Test acc   : {test_acc:.2f}%')
    print(f'  Eval time  : {elapsed:.2f}s ({len(test_dataset) / elapsed:.1f} images/s)')
    print('=' * 50)
    print()
    print('Confusion matrix (test set)')
    print('-' * 50)
    print_confusion_matrix(confusion, test_dataset.classes)


if __name__ == '__main__':
    main()


