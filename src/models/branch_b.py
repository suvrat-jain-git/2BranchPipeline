import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class BranchB(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        resnet_out_dim = cfg['model']['resnet_out_dim']   
        gru_hidden_dim = cfg['model']['gru_hidden_dim']   
        gru_layers = cfg['model']['gru_layers']          
        regressor_out_dim = cfg['model']['regressor_out_dim'] 
        smpl_num_tokens = cfg['model']['smpl_num_tokens']
        branch_b_out_dim = cfg['model']['branch_b_out_dim']

        # ResNet50 backbone — pretrained on ImageNet
        # classification head removed, only feature extractor kept
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # GRU
        self.gru = nn.GRU(
            input_size=resnet_out_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # SMPL parameter regressor
        # 512 => 82 (β + θ)
        self.regressor = nn.Linear(gru_hidden_dim, regressor_out_dim)

        # SMPL token encoder
        # expands 82-dim SMPL parameters to N tokens of 512-dim
        self.token_encoder = nn.Linear(
            regressor_out_dim,
            smpl_num_tokens * branch_b_out_dim
        )

        self.smpl_num_tokens = smpl_num_tokens
        self.branch_b_out_dim = branch_b_out_dim

    def forward(self, x):
        # x shape: [batch, T, 3, 224, 224]
        batch, T, C, H, W = x.shape

        # reshape to process all frames through ResNet50 in one pass
        # [batch, T, 3, 224, 224] => [batch*T, 3, 224, 224]
        x = x.view(batch * T, C, H, W)

        # apply ResNet50 to all frames with shared weights
        # [batch*T, 3, 224, 224] => [batch*T, 2048]
        x = self.backbone(x)

        # reshape back to separate batch and time dimensions
        # [batch*T, 2048] => [batch, T, 2048]
        x = x.view(batch, T, -1)

        # GRU processes sequence of T frame features
        # input:  [batch, T, 2048] => [batch, T, 512] 
        # output: x: all hidden state, _: final hidden state
        x, _ = self.gru(x)

        # take only the last time step hidden state
        # [batch, T, 512] => [batch, 512]
        x = x[:, -1, :]

        # regressor maps to SMPL parameters
        # [batch, 512] => [batch, 82]
        x = self.regressor(x)

        # SMPL token encoder expands to N tokens
        # [batch, 82] => [batch, N*512]
        x = self.token_encoder(x)

        # reshape to token sequence
        # [batch, N*512] => [batch, N, 512]
        x = x.view(batch, self.smpl_num_tokens, self.branch_b_out_dim)

        return x 
    
if __name__ == '__main__':
    import yaml

    with open('configs/smoke_test.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = BranchB(cfg)
    model.eval()

    # simulate a batch of 2 sequences, 8 frames each
    dummy_input = torch.randn(2, 8, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 8, 512])")
    print(f"Test passed:  {output.shape == torch.Size([2, 8, 512])}")