import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.data.dataset import CCVIDDataset
from src.models.pipeline import Pipeline
from collections import defaultdict

def freeze_backbone(model):
    for param in model.branch_a.backbone.parameters():
        param.requires_grad = False
    print("  Branch A frozen")

def unfreeze_backbone(model):
    for param in model.branch_a.backbone.parameters():
        param.requires_grad = True
    print("  Branch A unfrozen")

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_sequences_per_identity(dataset, val_ratio=0.2):
    import random
    identity_to_indices = defaultdict(list)
    for idx, (seq_path, identity) in enumerate(dataset.sequences):
        identity_to_indices[identity].append(idx)

    train_indices = []
    val_indices = []

    for identity, indices in identity_to_indices.items():
        indices = indices.copy()
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[-n_val:])
        train_indices.extend(indices[:-n_val])

    return train_indices, val_indices 

def get_optimizer(model, cfg):
    backbone_params = list(model.branch_a.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)

    new_params = [
        p for p in model.parameters()
        if p.requires_grad
        and id(p) not in backbone_ids
    ]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg['train']['backbone_lr']},
        {'params': new_params,      'lr': cfg['train']['base_lr']}
    ], weight_decay=cfg['train']['weight_decay'])

    return optimizer 

def get_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['train']['lr_step'],
        gamma=cfg['train']['lr_gamma']
    )

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(frames, labels)

        loss = criterion(output['identity'], labels)

        loss.backward()
        optimizer.step()

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

def compute_map(sim, query_labels, gallery_labels):
    ap_list = []

    for q in range(sim.shape[0]):
        q_label = query_labels[q]

        sorted_indices = sim[q].argsort(descending=True)
        sorted_labels = gallery_labels[sorted_indices]

        matches = (sorted_labels == q_label)

        if matches.sum() == 0:
            continue

        num_relevant = 0
        precision_sum = 0.0

        for rank, match in enumerate(matches):
            if match:
                num_relevant += 1
                precision_sum += num_relevant / (rank + 1)

        ap = precision_sum / matches.sum().item()
        ap_list.append(ap)

    return 100.0 * sum(ap_list) / len(ap_list) if ap_list else 0.0

def validate_one_epoch(model, val_loader, gallery_loader, device):
    model.eval()

    query_embeddings = []
    query_labels = []
    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device, non_blocking=True)
            output = model(frames, labels=None)
            emb = F.normalize(output['embedding'], dim=1)
            query_embeddings.append(emb.cpu())
            query_labels.append(labels)

    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad():
        for frames, labels in gallery_loader:
            frames = frames.to(device, non_blocking=True)
            output = model(frames, labels=None)
            emb = F.normalize(output['embedding'], dim=1)
            gallery_embeddings.append(emb.cpu())
            gallery_labels.append(labels)

    query_embeddings   = torch.cat(query_embeddings)    # [Nq, 512]
    query_labels       = torch.cat(query_labels)        # [Nq]
    gallery_embeddings = torch.cat(gallery_embeddings)  # [Ng, 512]
    gallery_labels     = torch.cat(gallery_labels)      # [Ng] 

    sim = torch.mm(query_embeddings, gallery_embeddings.t()) # [Nq, Ng]

    predicted = sim.argmax(dim=1)
    correct = (gallery_labels[predicted] == query_labels).sum().item()
    rank1 = 100.0 * correct / len(query_labels)

    map_score = compute_map(sim, query_labels, gallery_labels)

    return rank1, map_score

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

def save_best_checkpoint(model, optimizer, epoch, val_rank1, cfg):
    exp_name = cfg['train'].get('experiment_name', 'default')
    save_dir = f"runs/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_rank1': val_rank1
    }
    path = f"{save_dir}/best_checkpoint.pth"
    torch.save(checkpoint, path)
    print(f"  Best checkpoint saved: {path} "
          f"(epoch {epoch}, val_rank1={val_rank1:.2f}%)")

def train(cfg_path='configs/server.yaml'):
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'])
    print(f"\nDevice: {device}")

    print("\n[1/4] Loading dataset...")
    dataset = CCVIDDataset(cfg, split='train')
    print(f"  Using full dataset: {len(dataset)} sequences") 

    val_ratio = cfg['train'].get('val_ratio', 0.0)
    if val_ratio > 0:
        train_indices, val_indices = split_sequences_per_identity(dataset, val_ratio)
        train_dataset = Subset(dataset, train_indices)
        val_dataset   = Subset(dataset, val_indices)
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Val sequences:   {len(val_dataset)}")
    else:
        train_dataset = dataset
        val_dataset   = None
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  No validation split")

    pin_memory = (cfg['train']['device'] == 'cuda')

    loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=pin_memory
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers'],
            pin_memory=pin_memory
        )

    # gallery loader — train sequences as gallery during validation
    gallery_loader = None
    if val_dataset is not None:
        gallery_loader = DataLoader(
            train_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers'],
            pin_memory=pin_memory
        )

    print(f"  Train batches per epoch: {len(loader)}")

    print("\n[2/4] Building model...")
    model = Pipeline(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params:     {total:,}")

    # Phase 1 — freeze branch A
    print("\n  Phase 1: freezing branch A")
    freeze_backbone(model)
    print(f"  Trainable params: {count_trainable(model):,}")

    print("\n[3/4] Setting up optimizer...")
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()
    print(f"  Optimizer:    AdamW")
    print(f"  Base LR:      {cfg['train']['base_lr']}")
    print(f"  Backbone LR:  {cfg['train']['backbone_lr']}")
    print(f"  Weight decay: {cfg['train']['weight_decay']}")

    print("\n[4/4] Training...")
    num_epochs = cfg['train']['num_epochs']
    unfreeze_epoch = cfg['train']['unfreeze_all_epoch']
    val_every      = cfg['train'].get('val_every', 5)
    best_val_rank1 = 0.0
    best_val_map   = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        if epoch == unfreeze_epoch:
            print("  Phase 2:")
            unfreeze_backbone(model)
            print(f"  Trainable params: {count_trainable(model):,}")
            optimizer = get_optimizer(model, cfg)
            scheduler = get_scheduler(optimizer, cfg)
            print("  Optimizer and scheduler rebuilt for Phase 2")
            for i, group in enumerate(optimizer.param_groups):
                print(f"  Group {i} lr: {group['lr']}")

        avg_loss, accuracy = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch
        )

        scheduler.step()

        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Acc:  {accuracy:.2f}%")

        if val_loader is not None and epoch % val_every == 0:
            val_rank1, val_map = validate_one_epoch(
                model, val_loader, gallery_loader, device
            )
            print(f"  Val Rank-1: {val_rank1:.2f}%")
            print(f"  Val mAP:    {val_map:.2f}%")

            if val_rank1 > best_val_rank1:
                best_val_rank1 = val_rank1
                save_best_checkpoint(
                    model, optimizer, epoch, val_rank1, cfg
                )
            
            if val_map > best_val_map:
                best_val_map = val_map

        if epoch % cfg['train']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, cfg)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    train()
