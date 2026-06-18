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
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['train']['T_max'],
        eta_min=1e-6
    )

class IdentityBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, num_instances=4):
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids = batch_size // num_instances

        self.pid_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            if isinstance(dataset, Subset):
                real_idx = dataset.indices[idx]
                _, identity = dataset.dataset.sequences[real_idx]
            else:
                _, identity = dataset.sequences[idx]
            self.pid_to_indices[identity].append(idx)

        self.pids = list(self.pid_to_indices.keys())

    def __iter__(self):
        import random
        batch_indices = []
        pids = self.pids.copy()
        random.shuffle(pids)

        # iterate through all pids in groups of num_pids
        # each identity appears exactly once per epoch
        idx = 0
        while idx + self.num_pids <= len(pids):
            selected_pids = pids[idx:idx + self.num_pids]

            batch = []
            for pid in selected_pids:
                pid_indices = self.pid_to_indices[pid].copy()
                random.shuffle(pid_indices)
                while len(pid_indices) < self.num_instances:
                    pid_indices += self.pid_to_indices[pid].copy()
                batch.extend(pid_indices[:self.num_instances])

            batch_indices.extend(batch)
            idx += self.num_pids

        return iter(batch_indices)

    def __len__(self):
        num_batches = len(self.pids) // self.num_pids
        return num_batches * self.batch_size

def triplet_loss(embeddings, labels, margin=0.3):
    batch_size = embeddings.shape[0]

    # pairwise L2 distance matrix [batch, batch]
    dist = torch.cdist(embeddings, embeddings)

    # label masks [batch, batch]
    labels_col = labels.unsqueeze(0)   # [1, batch]
    labels_row = labels.unsqueeze(1)   # [batch, 1]

    pos_mask = (labels_row == labels_col)  # same identity
    neg_mask = (labels_row != labels_col)  # different identity

    # exclude diagonal (self distances)
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    pos_mask = pos_mask & ~eye

    # valid anchors — must have at least one positive and one negative
    valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # hardest positive — max distance among same-class
    # masked_fill sets non-positive pairs to -1e6 before max
    pos_dist = dist.masked_fill(~pos_mask, -1e6)
    hardest_pos = pos_dist.max(dim=1).values   # [batch]

    # hardest negative — min distance among different-class
    # masked_fill sets non-negative pairs to 1e6 before min
    neg_dist = dist.masked_fill(~neg_mask, 1e6)
    hardest_neg = neg_dist.min(dim=1).values   # [batch]

    # triplet loss per anchor
    triplet = F.relu(hardest_pos - hardest_neg + margin)

    # average only over valid anchors
    loss = triplet[valid].mean()

    return loss

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, cfg):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    triplet_margin = cfg['train'].get('triplet_margin', 0.3)
    triplet_weight = cfg['train'].get('triplet_weight', 0.5)

    for batch_idx, (frames, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(frames, labels)

        arcface_loss = criterion(output['identity'], labels)

        trip_loss = triplet_loss(
            output['embedding'],
            labels,
            margin=triplet_margin
        )

        loss = arcface_loss + triplet_weight * trip_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        predicted = output['identity'].argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"| ArcFace: {arcface_loss.item():.4f} "
                  f"| Triplet: {trip_loss.item():.4f}")
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
            query_embeddings.append(emb)
            query_labels.append(labels.to(device))

    gallery_embeddings = []
    gallery_labels = []
    with torch.no_grad():
        for frames, labels in gallery_loader:
            frames = frames.to(device, non_blocking=True)
            output = model(frames, labels=None)
            emb = F.normalize(output['embedding'], dim=1)
            gallery_embeddings.append(emb)
            gallery_labels.append(labels.to(device))

    query_embeddings   = torch.cat(query_embeddings)    # [Nq, 512]
    query_labels       = torch.cat(query_labels)        # [Nq]
    gallery_embeddings = torch.cat(gallery_embeddings)  # [Ng, 512]
    gallery_labels     = torch.cat(gallery_labels)      # [Ng] 

    sim = torch.mm(query_embeddings, gallery_embeddings.t()) # [Nq, Ng]

    predicted = sim.argmax(dim=1)
    correct = (gallery_labels[predicted] == query_labels).sum().item()
    rank1 = 100.0 * correct / len(query_labels)

    sim_cpu          = sim.cpu()
    query_labels_cpu = query_labels.cpu()
    gallery_labels_cpu = gallery_labels.cpu()

    map_score = compute_map(sim_cpu, query_labels_cpu, gallery_labels_cpu)

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

def save_best_checkpoint(model, optimizer, epoch, val_rank1, val_map, cfg):
    exp_name = cfg['train'].get('experiment_name', 'default')
    save_dir = f"runs/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_rank1': val_rank1,
        'val_map': val_map
    }
    path = f"{save_dir}/best_checkpoint.pth"
    torch.save(checkpoint, path)
    print(f"  Best checkpoint saved: {path} "
        f"(epoch {epoch}, "
        f"val_rank1={val_rank1:.2f}%, "
        f"val_map={val_map:.2f}%)")

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

    sampler = IdentityBalancedSampler(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        num_instances=cfg['train'].get('num_instances', 4)
    )

    loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['batch_size'],
        sampler=sampler,           # replaces shuffle=True
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
    print(f"  Triplet margin:   {cfg['train'].get('triplet_margin', 0.3)}")
    print(f"  Triplet weight:   {cfg['train'].get('triplet_weight', 0.5)}")

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
            remaining_epochs = num_epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                eta_min=1e-6
            )
            print("  Optimizer and scheduler rebuilt for Phase 2")
            print(f"  Remaining epochs: {remaining_epochs}")
            for i, group in enumerate(optimizer.param_groups):
                print(f"  Group {i} lr: {group['lr']}")

        avg_loss, accuracy = train_one_epoch(
            model, loader, optimizer, criterion, device, epoch, cfg
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

            if val_map >= best_val_map:
                best_val_map   = val_map
                best_val_rank1 = val_rank1
                save_best_checkpoint(
                    model, optimizer, epoch,
                    val_rank1, val_map, cfg
                )

        if epoch % cfg['train']['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, cfg)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    train()
