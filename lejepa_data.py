"""Multi-view CIFAR-10 dataset + augmentation for LeJEPA self-supervised
pretraining. Yields V augmented views /TRAIN unlabeled img
test images are never touched here
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Real CIFAR-10 per-channel mean/std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_ssl_transform():
    """SSL-style augmentation, toned down from ImageNet-scale SimCLR/LeJEPA
    settings to suit 32x32 CIFAR images:
    - RandomResizedCrop scale=(0.5, 1.0), looser than the reference repo's
      global/local multi-crop (0.3, 1.0)/(0.05, 0.3), since tighter crops
      on a 32x32 image shrink to patches too small to preserve object
      structure once resized back up.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


class MultiViewCIFAR10(Dataset):
    """Wraps CIFAR-10's TRAIN split & returns `views` independently
    augmented copies of each image, stacked as (V, C, H, W). Labels are
    discarded LeJEPA pretraining is fully unsupervised."""

    def __init__(self, root, views, transform=None):
        self.views = views
        self.transform = transform or build_ssl_transform()
        # train=True only,  pretraining must never see test images.
        self.base = CIFAR10(root=root, train=True, download=True, transform=None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]  # PIL image; label unused (unsupervised)
        views = [self.transform(img) for _ in range(self.views)]
        return torch.stack(views)  # (V, C, H, W)


def get_pretrain_loader(config):
    """Builds the multi-view unlabeled DataLoader
    Batches come out as (N, V, C, H, W) / collate_fn is
    the default (torch.stack over the dataset's (V,C,H,W) items), then we
    just rearrange the resulting (N,V,C,H,W) tensor to feed the encoder.
    """
    lejepa_cfg = config['lejepa']
    dataset = MultiViewCIFAR10(root=config['paths']['train_dir'], views=lejepa_cfg['views'])
    num_workers = lejepa_cfg['pretrain']['num_workers']
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=lejepa_cfg['pretrain']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,  # avoid respawning all workers every epoch
        drop_last=True,  # SIGReg's batch statistics are noisier on a ragged last batch
    )
    return loader
