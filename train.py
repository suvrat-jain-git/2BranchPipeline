import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.data.dataset import CCVIDDataset
from src.models.pipeline import Pipeline
from collections import defaultdict

def freeze_backbones(model):
    for param in model.branch_a.backbone.parameters():
        param.requires_grad = False
    if hasattr(model, 'branch_b'):
        if hasattr(model.branch_b, 'backbone'):
            for param in model.branch_b.backbone.parameters():
                param.requires_grad = False
        elif hasattr(model.branch_b, 'hmr2'):
            for param in model.branch_b.hmr2.parameters():
                param.requires_grad = False
        print("  Both backbones frozen")
    else:
        print("  Branch A backbone frozen (no Branch B)")

def unfreeze_hmr_backbone(model):
    if hasattr(model, 'branch_b'):
        if hasattr(model.branch_b, 'backbone'):
            for param in model.branch_b.backbone.parameters():
                param.requires_grad = True
            print("  Branch B backbone unfrozen (ResNet50)")
        elif hasattr(model.branch_b, 'hmr2'):
            print("  Branch B uses real HMR2.0 — keeping frozen (pretrained estimator)")
    else:
        print("  No Branch B to unfreeze — skipping")

def unfreeze_sapiens_backbone(model):
    for param in model.branch_a.backbone.parameters():
        param.requires_grad = True
    print("  Branch A backbone unfrozen")


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_by_identity(dataset, val_ratio=0.2):
    identity_to_indices = defaultdict(list)
    for idx, (seq_path, identity) in enumerate(dataset.sequences):
        identity_to_indices[identity].append(idx)

    train_indices = []
    val_indices = []

    for identity, indices in identity_to_indices.items():
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[-n_val:])
        train_indices.extend(indices[:-n_val])

    return train_indices, val_indices

def get_optimizer(model, cfg):
    backbone_params = list(model.branch_a.backbone.parameters())
    if hasattr(model, 'branch_b'):
        if hasattr(model.branch_b, 'backbone'):
            backbone_params += list(model.branch_b.backbone.parameters())
        elif hasattr(model.branch_b, 'hmr2'):
            backbone_params += list(model.branch_b.hmr2.parameters())

    backbone_ids = set(id(p) for p in backbone_params)

    new_params = [
        p for p in model.parameters()
        if id(p) not in backbone_ids
    ]

    optimizer = torch.optim.AdamW([
        {
            'params': backbone_params,
            'lr': cfg['train']['backbone_lr']
        },
        {
            'params': new_params,
            'lr': cfg['train']['base_lr']
        }
    ], weight_decay=cfg['train']['weight_decay'])

    return optimizer


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (frames, labels) in enumerate(loader):
        frames = frames.to(device)
        labels = labels.to(device)

        # zero gradients before forward pass
        optimizer.zero_grad()

        # forward pass
        output = model(frames, labels)

        # identity loss only — CCVID has no age/gender labels
        loss = criterion(output['identity'], labels)

        # backward pass
        loss.backward()
        optimizer.step()

        # track metrics
        total_loss += loss.item()
        predicted = output['identity'].argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}") 
            print(f"  Predictions: {predicted[:4].tolist()}")
            print(f"  Targets:     {labels[:4].tolist()}")


    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def validate_one_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            # inference mode — no labels passed to ArcFace
            output = model(frames, labels=None)

            # compute loss without ArcFace margin
            loss = criterion(output['identity'], labels)

            total_loss += loss.item()
            predicted = output['identity'].argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, cfg):
    exp_name = cfg['train'].get('experiment_name', 'default')
    save_dir = f"runs/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    path = f"{save_dir}/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")

def train(cfg_path='configs/server.yaml'):
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    # load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'])
    print(f"\nDevice: {device}")

    # Dataset 
    print("\n[1/4] Loading dataset...")
    dataset = CCVIDDataset(cfg, split='train')

    # use small subset locally for logic verification
    smoke_subset = cfg['train'].get('smoke_subset', 0)
    if smoke_subset > 0:
        dataset = Subset(dataset, list(range(smoke_subset)))
        print(f"  Using smoke subset: {smoke_subset} sequences")
    else:
        print(f"  Using full dataset: {len(dataset)} sequences")

    # split dataset into train and val
    val_ratio = cfg['train'].get('val_ratio', 0.0)
    if val_ratio > 0:
        train_indices, val_indices = split_by_identity(dataset, val_ratio)
        train_dataset = Subset(dataset, train_indices)
        val_dataset   = Subset(dataset, val_indices)
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Val sequences:   {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset   = None
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  No validation split")

    loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=False
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers'],
            pin_memory=False
        )

    print(f"  Train batches per epoch: {len(loader)}")

    # Model
    print("\n[2/4] Building model...")
    model = Pipeline(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params:     {total:,}")

    # Phase 1 — freeze both backbones
    print("\n  Phase 1: freezing both backbones")
    freeze_backbones(model)
    print(f"  Trainable params: {count_trainable(model):,}")

    # Optimizer and loss 
    print("\n[3/4] Setting up optimizer...")
    optimizer = get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['train']['lr_step'],
        gamma=cfg['train']['lr_gamma']
    )
    criterion = nn.CrossEntropyLoss()
    print(f"  Optimizer:    AdamW")
    print(f"  Base LR:      {cfg['train']['base_lr']}")
    print(f"  Backbone LR:  {cfg['train']['backbone_lr']}")
    print(f"  Weight decay: {cfg['train']['weight_decay']}")

    # Training loop
    print("\n[4/4] Training...")
    num_epochs = cfg['train']['num_epochs']
    unfreeze_hmr_epoch = cfg['train']['unfreeze_hmr_epoch']
    unfreeze_all_epoch = cfg['train']['unfreeze_all_epoch']

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Phase 2 — unfreeze Branch B backbone only
        if epoch == unfreeze_hmr_epoch:
            print("  Phase 2:")
            unfreeze_hmr_backbone(model)
            print(f"  Trainable params: {count_trainable(model):,}")
            # rebuild optimizer to include newly unfrozen params
            optimizer = get_optimizer(model, cfg)
            print("  Optimizer rebuilt for Phase 2")
            for i, group in enumerate(optimizer.param_groups):
                print(f"  Group {i} lr: {group['lr']}")

        # Phase 3 — unfreeze Branch A backbone
        if epoch == unfreeze_all_epoch:
            print("  Phase 3:")
            unfreeze_sapiens_backbone(model)
            print(f"  Trainable params: {count_trainable(model):,}")
            # rebuild optimizer to include newly unfrozen params
            optimizer = get_optimizer(model, cfg)
            print("  Optimizer rebuilt for Phase 3")
            for i, group in enumerate(optimizer.param_groups):
                print(f"  Group {i} lr: {group['lr']}")

        avg_loss, accuracy = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch
        )

        scheduler.step()

        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Acc:  {accuracy:.2f}%")

        # validation
        if val_loader is not None:
            val_loss, val_acc = validate_one_epoch(
                model, val_loader, criterion, device, epoch
            )
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.2f}%")

        if epoch % cfg['train']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, cfg)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    train()
