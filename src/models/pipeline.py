import torch
import torch.nn as nn
from src.models.branch_a import BranchA
from src.models.branch_b import BranchB
from src.models.fusion import Fusion
from src.models.heads import Heads


class Pipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.branch_a = BranchA(cfg)
        self.branch_b = BranchB(cfg)
        self.fusion = Fusion(cfg)
        self.heads = Heads(cfg)

    def forward(self, x, labels=None):
        # x: [batch, T, 3, 224, 224]

        # clothed body stream
        # [batch, T, 3, 224, 224] => [batch, T, 512]
        tokens_a = self.branch_a(x)

        # bare body stream
        # [batch, T, 3, 224, 224] => [batch, N, 512]
        tokens_b = self.branch_b(x)

        # cross attention fusion
        # [batch, T, 512] + [batch, N, 512] => [batch, 512]
        embedding = self.fusion(tokens_a, tokens_b)

        # task heads
        # [batch, 512] => {age, gender, identity}
        output = self.heads(embedding, labels)

        return output 
    
if __name__ == '__main__':
    import yaml

    with open('configs/smoke_test.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = Pipeline(cfg)

    dummy_input = torch.randn(2, 8, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1])

    # test training path
    model.train()
    output = model(dummy_input, dummy_labels)

    age_ok    = output['age'].shape      == torch.Size([2, 1])
    gender_ok = output['gender'].shape   == torch.Size([2, 1])
    id_ok     = output['identity'].shape == torch.Size([2, 75])

    print("Training mode:")
    print(f"  Input shape:    {dummy_input.shape}")
    print(f"  Age shape:      {output['age'].shape}      expected [2, 1]")
    print(f"  Gender shape:   {output['gender'].shape}      expected [2, 1]")
    print(f"  Identity shape: {output['identity'].shape}  expected [2, 75]")
    print(f"  Test passed:    {age_ok and gender_ok and id_ok}")

    # test inference path
    model.eval()
    with torch.no_grad():
        output = model(dummy_input, labels=None)

    age_ok    = output['age'].shape      == torch.Size([2, 1])
    gender_ok = output['gender'].shape   == torch.Size([2, 1])
    id_ok     = output['identity'].shape == torch.Size([2, 75])

    print("\nInference mode:")
    print(f"  Input shape:    {dummy_input.shape}")
    print(f"  Age shape:      {output['age'].shape}      expected [2, 1]")
    print(f"  Gender shape:   {output['gender'].shape}      expected [2, 1]")
    print(f"  Identity shape: {output['identity'].shape}  expected [2, 75]")
    print(f"  Test passed:    {age_ok and gender_ok and id_ok}")