import torch
import torch.nn as nn

class BranchA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        vit_out_dim = cfg['model']['vit_out_dim']
        temporal_heads = cfg['model']['temporal_heads']
        temporal_layers = cfg['model']['temporal_layers']
        temporal_dropout = cfg['model']['temporal_dropout']
        branch_a_out_dim = cfg['model']['branch_a_out_dim']
        num_frames = cfg['dataset']['num_frames']

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, vit_out_dim)
        ) # [1, T, 768]
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_out_dim,
            nhead=temporal_heads,
            dropout=temporal_dropout,
            batch_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers
        )

        self.proj = nn.Linear(vit_out_dim, branch_a_out_dim) 

        self.norm = nn.LayerNorm(branch_a_out_dim)

    def forward(self, x):
        # x shape: [batch, T, 3, 224, 224]
        batch, T, C, H, W = x.shape

        # [batch, T, 3, 224, 224] => [batch*T, 3, 224, 224]
        x = x.reshape(batch * T, C, H, W)

        # [batch*T, 3, 224, 224] => [batch*T, 768]
        x = self.backbone(x)

        # [batch*T, 768] => [batch, T, 768]
        x = x.reshape(batch, T, -1)

        # [batch, T, 768] + [1, T, 768]
        x = x + self.pos_embed[:, :T, :]

        # input:  [batch, T, 768] => [batch, T, 768] (time aware)
        x = self.temporal_transformer(x)

        # [batch, T, 768] => [batch, T, 512]
        x = self.proj(x)

        # [batch, T, 512] normalise
        x = self.norm(x)

        return x
