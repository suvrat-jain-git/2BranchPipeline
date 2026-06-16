import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceHead(nn.Module):
    def __init__(self, fused_dim, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes

        self.register_buffer('cos_m', torch.cos(torch.tensor(margin)))
        self.register_buffer('sin_m', torch.sin(torch.tensor(margin)))
        self.register_buffer('th', torch.cos(torch.tensor(torch.pi) - torch.tensor(margin)))
        self.register_buffer('mm', torch.sin(torch.tensor(torch.pi) - torch.tensor(margin)) * margin)

        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, fused_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        # x: [batch, 512]

        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)

        # output: [batch, num_classes]
        cosine = F.linear(x_norm, w_norm)

        if labels is None:
            # inference — return cosine scores directly
            return cosine
        
        # training — apply ArcFace margin
        # cos(theta + margin) = cos(theta)cos(margin) - sin(theta)sin(margin)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))

        phi = cosine * self.cos_m - sine * self.sin_m

        # numerical stability threshold
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # apply margin to target class only
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output = output * self.scale
        
        return output

class Heads(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        fused_dim = cfg['model']['fused_dim']        
        num_classes = cfg['model']['num_classes'] 

        self.dropout = nn.Dropout(cfg['model'].get('heads_dropout', 0.1))   

        self.identity_head = ArcFaceHead(fused_dim, num_classes)

    def forward(self, x, labels=None):
        # x: [batch, 512]

        embedding = F.normalize(x, dim=1)

        identity = self.identity_head(self.dropout(embedding), labels)

        return {
            'embedding': embedding, # [batch, 512]
            'identity': identity    # [batch, num_classes]
        } 
