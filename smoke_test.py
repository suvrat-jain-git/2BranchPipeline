import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.data.dataset import CCVIDDataset
from src.models.pipeline import Pipeline


def run_smoke_test(cfg_path='configs/smoke_test.yaml'):
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['smoke']['device'])
    print(f"\nDevice: {device}")

    # 1. Dataset 
    print("\n[1/5] Loading dataset...")
    dataset = CCVIDDataset(cfg, split='train')
    print(f"  Total sequences:    {len(dataset)}")
    print(f"  Unique identities:  {len(dataset.identity_to_idx)}")

    # take a tiny subset for smoke test
    num_sequences = cfg['smoke']['num_sequences']
    subset = Subset(dataset, list(range(num_sequences)))
    print(f"  Smoke test subset:  {num_sequences} sequences")

    loader = DataLoader(
        subset,
        batch_size=cfg['smoke']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    print(f"  Batches:            {len(loader)}")

    # 2. Model 
    print("\n[2/5] Building model...")
    model = Pipeline(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:      {total_params:,}")
    print(f"  Trainable params:  {trainable_params:,}")

    # 3. Forward pass
    print("\n[3/5] Running forward pass...")
    model.train()
    criterion = nn.CrossEntropyLoss()

    frames, labels = next(iter(loader))
    frames = frames.to(device)
    labels = labels.to(device)

    print(f"  Input frames shape: {frames.shape}")
    print(f"  Labels:             {labels.tolist()}")

    output = model(frames, labels)

    print(f"  Age shape:          {output['age'].shape}")
    print(f"  Gender shape:       {output['gender'].shape}")
    print(f"  Identity shape:     {output['identity'].shape}")

    # 4. Loss computation
    print("\n[4/5] Computing loss...")
    loss = criterion(output['identity'], labels)
    print(f"  Identity loss:      {loss.item():.4f}")
    print(f"  Loss is finite:     {torch.isfinite(loss).item()}")

    # 5. Backward pass
    print("\n[5/5] Running backward pass...")
    loss.backward() 
    
    print("  Backward pass:      OK")

    # check gradients are flowing
    grad_ok = all(
        p.grad is not None
        for name, p in model.named_parameters()
        if p.requires_grad
        and 'age_head' not in name
        and 'gender_head' not in name
    )

    print(f"  Gradients flowing:  {grad_ok}")

    # Result 
    print("\n" + "=" * 60)
    all_ok = torch.isfinite(loss).item() and grad_ok
    print(f"SMOKE TEST {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 60)


if __name__ == '__main__':
    run_smoke_test()