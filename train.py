import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.data.dataset import CCVIDDataset
from src.models.pipeline import Pipeline


def freeze_backbones(model):
    for param in model.branch_a.backbone.parameters():
        param.requires_grad = False
    for param in model.branch_b.backbone.parameters():
        param.requires_grad = False
    print("  Both backbones frozen")


def unfreeze_hmr_backbone(model):
    for param in model.branch_b.backbone.parameters():
        param.requires_grad = True
    print("  Branch B backbone unfrozen")


def unfreeze_sapiens_backbone(model):
    for param in model.branch_a.backbone.parameters():
        param.requires_grad = True
    print("  Branch A backbone unfrozen")


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model, cfg):
    # backbone params — pretrained, use lower lr
    backbone_params = (
        list(model.branch_a.backbone.parameters()) +
        list(model.branch_b.backbone.parameters())
    )
    backbone_ids = set(id(p) for p in backbone_params)

    # new components — randomly initialised, use higher lr
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


def save_checkpoint(model, optimizer, epoch, loss, cfg):
    os.makedirs('runs', exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    path = f"runs/checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def train(cfg_path='configs/smoke_test.yaml'):
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

    loader = DataLoader(
        dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=False
    )
    print(f"  Batches per epoch: {len(loader)}")

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

        print(f"  Avg Loss:  {avg_loss:.4f}")
        print(f"  Accuracy:  {accuracy:.2f}%")

        if epoch % cfg['train']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, cfg)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    train()