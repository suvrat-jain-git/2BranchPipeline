import torch
import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class BranchB(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # resnet_out_dim = cfg['model']['resnet_out_dim']
        hmr2_out_dim = cfg['model']['hmr2_out_dim']
        gru_layers = cfg['model']['gru_layers']
        smpl_num_tokens = cfg['model']['smpl_num_tokens']
        branch_b_out_dim = cfg['model']['branch_b_out_dim']

        # ResNet50 backbone — pretrained on ImageNet
        # classification head removed, only feature extractor kept
        # backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # backbone.fc = nn.Identity()
        # self.backbone = backbone

	# mock HMR SMPL head - give SMPL parameters
        # self.mock_smpl_head = nn.Linear(resnet_out_dim,hmr2_out_dim)

        # HMR2.0 — pretrained human mesh recovery network
        # produces real β (shape) and θ (pose) parameters per frame
        # frozen permanently — used as a fixed bare body estimator
        from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
        self.hmr2, self.hmr2_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.hmr2 = self.hmr2.cuda()
        self.hmr2.eval()
        for param in self.hmr2.parameters():
            param.requires_grad = False
	
        # GRU — temporal consistency across T frames
        # HMR2.0 processes each frame independently
        # GRU sees all T frames in sequence and produces
        # a temporally consistent SMPL parameter summary
        # input:  226-dim SMPL params per frame
        # output: 226-dim temporally consistent SMPL params
        self.gru = nn.GRU(
            input_size=hmr2_out_dim,
            hidden_size=hmr2_out_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # SMPL token encoder
        # expands 82-dim SMPL parameters to N tokens of 512-dim
        self.token_encoder = nn.Linear(
            hmr2_out_dim,
            smpl_num_tokens * branch_b_out_dim
        )

        self.smpl_num_tokens = smpl_num_tokens
        self.branch_b_out_dim = branch_b_out_dim
        self.hmr2_out_dim = hmr2_out_dim 

    def forward(self, x):
        # x shape: [batch, T, 3, 224, 224]
        batch, T, C, H, W = x.shape

        # reshape to process all frames through ResNet50 in one pass
        # [batch, T, 3, 224, 224] => [batch*T, 3, 224, 224]
        x = x.view(batch * T, C, H, W)

        # apply ResNet50 to all frames with shared weights
        # [batch*T, 3, 224, 224] => [batch*T, 2048]
        # x = self.backbone(x)

	# mock HMR SMPL head predicts SMPL parameters
	# [batch*T, 2048] => [batch*T, 82]
        # x = self.mock_smpl_head(x)

        # resize to 256x256 for HMR2.0
        # HMR2.0 internally crops x[:,:,:,32:-32] => 256x192
        # Branch A keeps 224x224 for DINOv2 (needs multiple of 14)
        x_resized = F.interpolate(
            x,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        # HMR2.0 forward pass (frozen, no gradients needed)
        # produces real SMPL parameters per frame
        with torch.no_grad():
            out = self.hmr2({'img': x_resized})

        # extract SMPL parameters and flatten to vectors
        betas = out['pred_smpl_params']['betas']                                      # [batch*T, 10]
        global_orient = out['pred_smpl_params']['global_orient'].reshape(batch * T, -1)  # [batch*T, 9]
        body_pose = out['pred_smpl_params']['body_pose'].reshape(batch * T, -1)          # [batch*T, 207]

        # concatenate all SMPL parameters
        # [batch*T, 10] + [batch*T, 9] + [batch*T, 207] => [batch*T, 226]
        smpl_params = torch.cat([betas, global_orient, body_pose], dim=1)

        # move to same device as GRU parameters
        device = next(self.gru.parameters()).device
        smpl_params = smpl_params.to(device)

        # reshape to sequence for GRU
        # [batch*T, 226] => [batch, T, 226]
        smpl_params = smpl_params.view(batch, T, self.hmr2_out_dim)

        # reshape back to separate batch and time dimensions
        # [batch*T, 82] => [batch, T, 82]
        # x = x.view(batch, T, -1)

        # GRU processes sequence of T frame features
        # input:  [batch, T, 226] => [batch, T,226]
        # output: x: all hidden state, _: final hidden state
        x, _ = self.gru(smpl_params)

        # take only the last time step hidden state
        # [batch, T, 226] => [batch, 226]
        x = x[:, -1, :]

        # SMPL token encoder expands to N tokens
        # [batch, 226] => [batch, N*512]
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
    dummy_input = torch.randn(2, 8, 3, 224, 224).cuda()

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 8, 512])")
    print(f"Test passed:  {output.shape == torch.Size([2, 8, 512])}")
