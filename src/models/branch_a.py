import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class BranchA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        vit_out_dim = cfg['model']['vit_out_dim']
        temporal_heads = cfg['model']['temporal_heads']     
        temporal_layers = cfg['model']['temporal_layers']   
        temporal_dropout = cfg['model']['temporal_dropout'] 
        branch_a_out_dim = cfg['model']['branch_a_out_dim'] 

        # ViT-B backbone — pretrained on ImageNet
        # classification head removed, only encoder kept
        backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        backbone.heads = nn.Identity()
        self.backbone = backbone

        # temporal transformer
        # input shape: [batch, seq, dim]
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

        # linear projection
        self.proj = nn.Linear(vit_out_dim, branch_a_out_dim)

    def forward(self, x):
        # x shape: [batch, T, 3, 224, 224]
        batch, T, C, H, W = x.shape

        # reshape to process all frames through ViT-B in one pass
        # [batch, T, 3, 224, 224] => [batch*T, 3, 224, 224]
        x = x.view(batch * T, C, H, W)

        # apply ViT-B to all frames with shared weights
        # [batch*T, 3, 224, 224] => [batch*T, 768]
        x = self.backbone(x)

        # reshape back to separate batch and time dimensions
        # [batch*T, 768] => [batch, T, 768]
        x = x.view(batch, T, -1)

        # temporal transformer models relationships across T frames
        # input:  [batch, T, 768] => [batch, T, 768] (time aware)
        x = self.temporal_transformer(x)

        # linear projection to common fusion dimension
        # [batch, T, 768] => [batch, T, 512]
        x = self.proj(x)

        return x 
    
if __name__ == '__main__':
    import yaml

    with open('configs/smoke_test.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = BranchA(cfg)
    model.eval()

    # simulate a batch of 2 sequences, 8 frames each
    dummy_input = torch.randn(2, 8, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 8, 512])")
    print(f"Test passed:  {output.shape == torch.Size([2, 8, 512])}")