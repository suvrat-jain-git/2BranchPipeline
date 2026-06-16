import torch
import torch.nn as nn
import torch.nn.functional as F

class BranchB(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        hmr2_out_dim = cfg['model']['hmr2_out_dim']
        gru_layers = cfg['model']['gru_layers']
        branch_b_out_dim = cfg['model']['branch_b_out_dim']
        dropout = cfg['model'].get('branch_b_dropout', 0.1)

        # frozen permanently — used as a fixed bare body estimator
        from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
        self.hmr2, self.hmr2_cfg = load_hmr2(DEFAULT_CHECKPOINT)
        self.hmr2.eval()
        for param in self.hmr2.parameters():
            param.requires_grad = False

        self.gru = nn.GRU(
            input_size=hmr2_out_dim,
            hidden_size=hmr2_out_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(dropout)

        self.token_encoder = nn.Linear(hmr2_out_dim, branch_b_out_dim)

        self.norm = nn.LayerNorm(branch_b_out_dim)

        self.branch_b_out_dim = branch_b_out_dim
        self.hmr2_out_dim = hmr2_out_dim 

    def train(self, mode=True):
        super().train(mode)
        self.hmr2.eval()
        return self

    def forward(self, x):
        # x shape: [batch, T, 3, 224, 224]
        batch, T, C, H, W = x.shape

        # [batch, T, 3, 224, 224] => [batch*T, 3, 224, 224]
        x = x.reshape(batch * T, C, H, W)

        # [batch*T, 3, 224, 224] => [batch*T, 3, 256, 256]
        x_resized = F.interpolate(
            x,
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )

        with torch.no_grad():
            out = self.hmr2({'img': x_resized})

        betas = out['pred_smpl_params']['betas']                                         # [batch*T, 10]
        global_orient = out['pred_smpl_params']['global_orient'].reshape(batch * T, -1)  # [batch*T, 9]
        body_pose = out['pred_smpl_params']['body_pose'].reshape(batch * T, -1)          # [batch*T, 207]

        # [batch*T, 10] + [batch*T, 9] + [batch*T, 207] => [batch*T, 226]
        smpl_params = torch.cat([betas, global_orient, body_pose], dim=1)

        # [batch*T, 226] => [batch, T, 226]
        smpl_params = smpl_params.reshape(batch, T, self.hmr2_out_dim)

        # [batch, T, 226] => [batch, T, 226]
        x, _ = self.gru(smpl_params)

        x = self.dropout(x)

        # [batch, T, 226] => [batch, T, 512]
        x = self.token_encoder(x)

        # [batch, T, 512] normalise
        x = self.norm(x)

        return x
