import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from src.models.branch_a import BranchA
from src.models.branch_b import BranchB
from src.models.fusion import Fusion
from src.models.heads import Heads


class Pipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.use_branch_b = cfg['model'].get('use_branch_b', True)

        self.branch_a = BranchA(cfg)

        if self.use_branch_b:
            self.branch_b = BranchB(cfg)
            self.fusion = Fusion(cfg)
        else:
            # branch A only — project directly to fused_dim
            branch_a_out_dim = cfg['model']['branch_a_out_dim']
            fused_dim = cfg['model']['fused_dim']
            self.proj = nn.Linear(branch_a_out_dim, fused_dim)
            self.norm = nn.LayerNorm(fused_dim)

        self.heads = Heads(cfg)

    def forward(self, x, labels=None):
        # x: [batch, T, 3, 224, 224]

        # clothed body stream
        tokens_a = self.branch_a(x)

        if self.use_branch_b:
            # bare body stream
            tokens_b = self.branch_b(x)

            # cross attention fusion
            # [batch, T, 512] + [batch, N, 512] => [batch, 512]
            embedding = self.fusion(tokens_a, tokens_b)
        else:
            # branch A only
            # average pool across T time steps
            # [batch, T, 512] => [batch, 512]
            embedding = tokens_a.mean(dim=1)
            embedding = self.proj(embedding)
            embedding = self.norm(embedding)

        # task heads
        output = self.heads(embedding, labels)

        return output

if __name__ == '__main__':
    import yaml

    with open('configs/server.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    dummy_input = torch.randn(2, 8, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1])

    for use_branch_b in [True, False]:
        cfg['model']['use_branch_b'] = use_branch_b
        model = Pipeline(cfg)

        mode_name = "Full pipeline" if use_branch_b else "Branch A only"
        print(f"\n{'='*50}")
        print(f"{mode_name} (use_branch_b={use_branch_b})")
        print(f"{'='*50}")

        # test training path
        model.train()
        output = model(dummy_input, dummy_labels)

        age_ok    = output['age'].shape      == torch.Size([2, 1])
        gender_ok = output['gender'].shape   == torch.Size([2, 1])
        id_ok     = output['identity'].shape == torch.Size([2, 75])

        print("Training mode:")
        print(f"  Age shape:      {output['age'].shape}      expected [2, 1]")
        print(f"  Gender shape:   {output['gender'].shape}      expected [2, 1]")
        print(f"  Identity shape: {output['identity'].shape}  expected [2, 75]")
        print(f"  Test passed:    {age_ok and gender_ok and id_ok}")

        # test inference path
        model.eval()
        with torch.no_grad():
            output = model(dummy_input, labels=None)

        age_ok    = output['age'].shape      == torch.Size([2, 1])
        gender_ok = output['gender'].shape   == torch.Size([2, 1])
        id_ok     = output['identity'].shape == torch.Size([2, 75])

        print("Inference mode:")
        print(f"  Age shape:      {output['age'].shape}      expected [2, 1]")
        print(f"  Gender shape:   {output['gender'].shape}      expected [2, 1]")
        print(f"  Identity shape: {output['identity'].shape}  expected [2, 75]")
        print(f"  Test passed:    {age_ok and gender_ok and id_ok}")
