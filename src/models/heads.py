import torch
import torch.nn as nn
import torch.nn.functional as F

# note: CCVID dataset has no age and gender labels
class AgeHead(nn.Module):
    def __init__(self, fused_dim):
        super().__init__()
        self.fc = nn.Linear(fused_dim, 1)

    def forward(self, x):
        # x: [batch, 512]
        # output: [batch, 1]
        return self.fc(x)


class GenderHead(nn.Module):
    def __init__(self, fused_dim):
        super().__init__()
        self.fc = nn.Linear(fused_dim, 1)

    def forward(self, x):
        # x: [batch, 512]
        # output: [batch, 1] 
        # note: sigmoid applied during loss computation
        return self.fc(x)


class ArcFaceHead(nn.Module):
    def __init__(self, fused_dim, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        # learnable class centres in embedding space
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, fused_dim)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        # x: [batch, 512]

        # l2 normalise embeddings and class centres
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)

        # cosine similarity between embeddings and all class centres
        # output: [batch, num_classes]
        cosine = F.linear(x_norm, w_norm)

        if labels is None:
            # testing — return cosine scores directly
            return cosine

        # training — apply ArcFace margin
        # add angular margin to the target class only
        # for wrong class: cos(theta) while correct class: cos(theta+margin)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = torch.cos(theta + self.margin * one_hot)
        output = output * self.scale

        return output


class Heads(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        fused_dim = cfg['model']['fused_dim']        
        num_classes = cfg['model']['num_classes']    

        self.age_head = AgeHead(fused_dim)
        self.gender_head = GenderHead(fused_dim)
        self.identity_head = ArcFaceHead(fused_dim, num_classes)

    def forward(self, x, labels=None):
        # x: [batch, 512]

        age = self.age_head(x)          
        gender = self.gender_head(x)
        identity = self.identity_head(x, labels)

        return {
            'age': age,
            'gender': gender,
            'identity': identity
        } 
    
if __name__ == '__main__':
    import yaml

    with open('configs/smoke_test.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = Heads(cfg)

    dummy_embedding = torch.randn(2, 512)
    dummy_labels = torch.tensor([0, 1])

    # test training path
    model.train()
    output_train = model(dummy_embedding, dummy_labels)

    print("Training mode:")
    print(f"  Input shape:     {dummy_embedding.shape}")
    print(f"  Age shape:       {output_train['age'].shape}")
    print(f"  Gender shape:    {output_train['gender'].shape}")
    print(f"  Identity shape:  {output_train['identity'].shape}")
    print(f"  Expected:       [2,1], [2,1], [2,75]")
    print(f"  Age passed:      {output_train['age'].shape == torch.Size([2, 1])}")
    print(f"  Gender passed:   {output_train['gender'].shape == torch.Size([2, 1])}")
    print(f"  Identity passed: {output_train['identity'].shape == torch.Size([2, 75])}")

    # test inference path
    model.eval()
    with torch.no_grad():
        output_infer = model(dummy_embedding, labels=None)

    print("\nInference mode (no labels):")
    print(f"  Input shape:     {dummy_embedding.shape}")
    print(f"  Age shape:       {output_infer['age'].shape}")
    print(f"  Gender shape:    {output_infer['gender'].shape}")
    print(f"  Identity shape:  {output_infer['identity'].shape}")
    print(f"  Expected:       [2,1], [2,1], [2,75]")
    print(f"  Age passed:      {output_infer['age'].shape == torch.Size([2, 1])}")
    print(f"  Gender passed:   {output_infer['gender'].shape == torch.Size([2, 1])}")
    print(f"  Identity passed: {output_infer['identity'].shape == torch.Size([2, 75])}")