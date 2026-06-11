import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        branch_a_out_dim = cfg['model']['branch_a_out_dim']  
        branch_b_out_dim = cfg['model']['branch_b_out_dim']  
        fusion_heads = cfg['model']['fusion_heads']          
        fused_dim = cfg['model']['fused_dim']              

        # cross attention
        # Q = Branch A clothed tokens
        # K,V = Branch B bare body tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=branch_a_out_dim,
            num_heads=fusion_heads,
            batch_first=True
        )

        # linear projection after average pooling
        self.linear = nn.Linear(branch_a_out_dim, fused_dim)

        # layer norm for stable scale before task heads
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, tokens_a, tokens_b):
        # tokens_a: [batch, T, 512]   Q — clothed body
        # tokens_b: [batch, N, 512]   K,V — bare body

        # cross attention
        # Q = tokens_a, K = tokens_b, V = tokens_b
        # output: [batch, T, 512]
        # fused: fused tokens, _: attention weights
        fused, _ = self.cross_attention(
            query=tokens_a,
            key=tokens_b,
            value=tokens_b
        )

        # average pool across T time steps
        # [batch, T, 512] => [batch, 512]
        fused = fused.mean(dim=1)

        # linear transformation
        # [batch, 512] => [batch, 512]
        fused = self.linear(fused)

        # layer norm
        # [batch, 512] => [batch, 512]
        fused = self.norm(fused)

        return fused 

if __name__ == '__main__':
    import yaml

    with open('configs/smoke_test.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = Fusion(cfg)
    model.eval()

    tokens_a = torch.randn(2, 8, 512)
    tokens_b = torch.randn(2, 8, 512)

    with torch.no_grad():
        output = model(tokens_a, tokens_b)

    print(f"tokens_a shape: {tokens_a.shape}")
    print(f"tokens_b shape: {tokens_b.shape}")
    print(f"Output shape:   {output.shape}")
    print(f"Expected:       torch.Size([2, 512])")
    print(f"Test passed:    {output.shape == torch.Size([2, 512])}")