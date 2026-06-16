import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        branch_a_out_dim = cfg['model']['branch_a_out_dim']  
        branch_b_out_dim = cfg['model']['branch_b_out_dim']  
        fusion_heads = cfg['model']['fusion_heads']          
        dropout = cfg['model'].get('fusion_dropout', 0.1)          

        assert branch_a_out_dim == branch_b_out_dim, \
            f"branch_a_out_dim ({branch_a_out_dim}) must equal branch_b_out_dim ({branch_b_out_dim})"
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=branch_a_out_dim,
            num_heads=fusion_heads,
            batch_first=True,
            dropout=dropout
        )

        self.token_norm = nn.LayerNorm(branch_a_out_dim)

        self.mlp = nn.Sequential(
            nn.Linear(branch_a_out_dim, branch_a_out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_a_out_dim * 4, branch_a_out_dim)
        )

        self.token_norm2 = nn.LayerNorm(branch_a_out_dim)

        self.norm = nn.LayerNorm(branch_a_out_dim)

    def forward(self, tokens_a, tokens_b):
        # tokens_a: [batch, T, 512]   
        # tokens_b: [batch, T, 512]   

        # Q = tokens_a, K = tokens_b, V = tokens_b
        # attn_out:     [batch, T, 512]
        # attn_weights: [batch, T, T]
        attn_out, attn_weights = self.cross_attention(
            query=tokens_a,
            key=tokens_b,
            value=tokens_b
        )

        # [batch, T, 512]
        fused = tokens_a + attn_out

        fused = self.token_norm(fused) 

        # [batch, T, 512] => [batch, T, 512]
        mlp_out = self.mlp(fused)
        fused = fused + mlp_out 

        fused = self.token_norm2(fused)

        # [batch, T, 512] => [batch, 512]
        fused = fused.mean(dim=1)

        # [batch, 512] => [batch, 512]
        fused = self.norm(fused)

        return fused 
    