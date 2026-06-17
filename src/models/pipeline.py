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
            self.norm = nn.LayerNorm(cfg['model']['branch_a_out_dim'])

        self.heads = Heads(cfg)

    def forward(self, x, labels=None):
        # x: [batch, T, 3, 224, 224]

        # clothed body stream
        tokens_a = self.branch_a(x)

        if self.use_branch_b:
            # bare body stream
            tokens_b = self.branch_b(x)
            embedding = self.fusion(tokens_a, tokens_b)
        else:
            # average pool across T time steps
            # [batch, T, 512] => [batch, 512]
            embedding = tokens_a.mean(dim=1)
            embedding = self.norm(embedding)

        return self.heads(embedding, labels)  