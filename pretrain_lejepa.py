"""LeJEPA self-supervised pretraining on CIFAR-10 (train split only)
wires up model.py's CIFARResNetEncoder, lejepa_data.py's multi-view loader,& lejepa_loss.py's LeJEPALoss into a training loop 
No labels are used at this stage (later extended for fine-tuning) for the supervised path
"""
import math
import os

import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from lejepa_data import get_pretrain_loader
from lejepa_loss import LeJEPALoss
from model import CIFARResNetEncoder

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def build_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs, min_lr_ratio=1e-3):
    """Linear warmup for `warmup_epochs`, then cosine annealing down to
    `min_lr_ratio * initial_lr` for the remaining epochs. no weight-decay schedule"""
    warmup_steps = steps_per_epoch * warmup_epochs
    total_steps = steps_per_epoch * epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_resume_checkpoint(resume_path, epoch, encoder, optimizer, scheduler):
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, resume_path)


def pretrain(encoder, dataloader, loss_fn, optimizer, scheduler, device, epochs, resume_path, start_epoch=0):
    encoder.train()
    for epoch in range(start_epoch, epochs):
        running_total, running_pred, running_sigreg = 0.0, 0.0, 0.0
        with tqdm(
            dataloader,
            desc=f"Pretrain epoch {epoch+1}/{epochs}",
            leave=True,
            unit="batch"
        ) as progress_bar:
            for i, views in enumerate(progress_bar):
                views = views.to(device, non_blocking=True)  # (N, V, C, H, W)

                optimizer.zero_grad()
                _, proj = encoder(views)  # proj: (V, N, proj_dim)
                total_loss, pred_loss, sigreg_loss = loss_fn(proj)
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                running_total += total_loss.item()
                running_pred += pred_loss.item()
                running_sigreg += sigreg_loss.item()
                progress_bar.set_postfix({
                    'loss': running_total / (i + 1),
                    'pred': running_pred / (i + 1),
                    'sigreg': running_sigreg / (i + 1),
                })

        n_batches = len(dataloader)
        print(
            f"Epoch {epoch+1} finished. "
            f"Total loss: {running_total / n_batches:.4f} | "
            f"Pred loss: {running_pred / n_batches:.4f} | "
            f"SIGReg loss: {running_sigreg / n_batches:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        save_resume_checkpoint(resume_path, epoch, encoder, optimizer, scheduler)
        print(f"Resume checkpoint saved to {resume_path} (epoch {epoch+1})")
    print('Finished LeJEPA pretraining')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    lejepa_cfg = config['lejepa']
    pretrain_cfg = lejepa_cfg['pretrain']

    dataloader = get_pretrain_loader(config)

    encoder = CIFARResNetEncoder(
        embedding_dim=lejepa_cfg['embedding_dim'],
        proj_dim=lejepa_cfg['proj_dim'],
    ).to(device)

    loss_fn = LeJEPALoss(lam=lejepa_cfg['lambda'], num_slices=lejepa_cfg['num_slices']).to(device)

    optimizer = optim.AdamW(
        encoder.parameters(),
        lr=pretrain_cfg['lr'],
        weight_decay=pretrain_cfg['weight_decay'],
    )
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=len(dataloader),
        epochs=pretrain_cfg['epochs'],
        warmup_epochs=pretrain_cfg['warmup_epochs'],
    )

    resume_path = config['paths']['lejepa_pretrain_resume_path']
    start_epoch = 0
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from {resume_path} at epoch {start_epoch + 1}")

    pretrain(encoder, dataloader, loss_fn, optimizer, scheduler, device, pretrain_cfg['epochs'], resume_path, start_epoch)

    os.makedirs(os.path.dirname(config['paths']['lejepa_encoder_path']), exist_ok=True)
    torch.save(encoder.state_dict(), config['paths']['lejepa_encoder_path'])
    print(f"Encoder saved to {config['paths']['lejepa_encoder_path']}")


if __name__ == '__main__':
    main()
