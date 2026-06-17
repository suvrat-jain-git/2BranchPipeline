import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.data.dataset import CCVIDDataset
from src.models.pipeline import Pipeline

def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for frames, label in loader:
            frames = frames.to(device, non_blocking=True)
            output = model(frames, labels=None)
            emb = F.normalize(output['embedding'], dim=1)
            embeddings.append(emb.cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)  # total_sequences x 512
    labels = torch.cat(labels, dim=0)          # total_sequences

    return embeddings, labels 
 
def compute_rank_metrics(query_emb, query_labels,
                        gallery_emb, gallery_labels, ranks=(1,5,10)):
    sim_matrix = torch.mm(query_emb, gallery_emb.t())
    num_query = query_emb.shape[0]
    rank_correct = {r: 0 for r in ranks}

    for i in range(num_query):
        sim = sim_matrix[i]
        q_label = query_labels[i].item()

        sorted_indices = sim.argsort(descending=True)
        sorted_labels = gallery_labels[sorted_indices]

        for r in ranks:
            if q_label in sorted_labels[:r].tolist():
                rank_correct[r] += 1

    rank_acc = {r: 100.0 * rank_correct[r] / num_query for r in ranks}
    return rank_acc

def compute_map(query_emb, query_labels,
                gallery_emb, gallery_labels):
    
    sim_matrix = torch.mm(query_emb, gallery_emb.t()) 
    num_query = query_emb.shape[0]  
    average_precisions = []

    for i in range(num_query):
        sim = sim_matrix[i]
        q_label = query_labels[i].item()

        sorted_indices = sim.argsort(descending=True)
        sorted_labels = gallery_labels[sorted_indices]

        correct_mask = (sorted_labels == q_label)
        num_correct = correct_mask.sum().item()

        if num_correct == 0:
            continue

        positions = torch.where(correct_mask)[0].float() + 1  # 1-indexed
        precisions = torch.arange(1, num_correct + 1).float() / positions
        ap = precisions.mean().item()
        average_precisions.append(ap)

    if len(average_precisions) == 0:
        return 0.0

    return 100.0 * sum(average_precisions) / len(average_precisions)


def compute_eer(query_emb, query_labels,
                gallery_emb, gallery_labels,
                max_query=1000):
    num_query = query_emb.shape[0]

    if num_query > max_query:
        print(f"  Skipping EER — {num_query} queries exceed max_query={max_query}")
        return -1.0

    sim_matrix = torch.mm(query_emb, gallery_emb.t())
    num_gallery = gallery_emb.shape[0]
    
    scores = []
    is_genuine = []

    for i in range(num_query):
        for j in range(num_gallery):
            score = sim_matrix[i, j].item()
            genuine = (query_labels[i].item() == gallery_labels[j].item())
            scores.append(score)
            is_genuine.append(int(genuine))

    scores = torch.tensor(scores)
    is_genuine = torch.tensor(is_genuine)

    thresholds = torch.linspace(-1, 1, steps=1000)
    min_diff = float('inf')
    eer = 0.0

    for threshold in thresholds:
        predicted_genuine = (scores >= threshold)
        genuine_mask  = (is_genuine == 1)
        impostor_mask = (is_genuine == 0)

        # FAR = false accepts / total impostors
        far = (predicted_genuine & impostor_mask).sum().float() / \
               impostor_mask.sum().float()

        # FRR = false rejects / total genuines
        frr = (~predicted_genuine & genuine_mask).sum().float() / \
               genuine_mask.sum().float()

        diff = abs(far.item() - frr.item())
        if diff < min_diff:
            min_diff = diff
            eer = (far.item() + frr.item()) / 2.0

    return eer * 100.0

def evaluate(cfg_path='configs/server.yaml',
             checkpoint_path=None):
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['evaluate']['device'])
    print(f"\nDevice: {device}")

    print("\n[1/4] Loading model...")
    model = Pipeline(cfg).to(device)

    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f"  Loaded checkpoint from epoch {epoch}")
    else:
        print("  No checkpoint provided — using random weights")
        print("  (for pipeline verification only)")

    print("\n[2/4] Loading gallery and query datasets...")
    gallery_dataset = CCVIDDataset(cfg, split='gallery')
    query_dataset   = CCVIDDataset(cfg, split='query')

    gallery_subset = cfg['evaluate'].get('gallery_subset', 0)
    query_subset   = cfg['evaluate'].get('query_subset', 0)

    if gallery_subset > 0:
        gallery_dataset = Subset(
            gallery_dataset,
            list(range(gallery_subset))
        )
        print(f"  Using gallery subset: {gallery_subset} sequences")
    else:
        print(f"  Using full gallery:   {len(gallery_dataset)} sequences")

    if query_subset > 0:
        query_dataset = Subset(
            query_dataset,
            list(range(query_subset))
        )
        print(f"  Using query subset:   {query_subset} sequences")
    else:
        print(f"  Using full query:     {len(query_dataset)} sequences")

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg['evaluate']['batch_size'],
        shuffle=False,
        num_workers=cfg['evaluate']['num_workers'],
        pin_memory=(cfg['evaluate']['device'] == 'cuda')
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg['evaluate']['batch_size'],
        shuffle=False,
        num_workers=cfg['evaluate']['num_workers'],
        pin_memory=(cfg['evaluate']['device'] == 'cuda')
    )

    print(f"  Gallery sequences: {len(gallery_dataset)}")
    print(f"  Query sequences:   {len(query_dataset)}")

    print("\n[3/4] Extracting embeddings...")
    gallery_emb, gallery_labels = extract_embeddings(
        model, gallery_loader, device
    )
    query_emb, query_labels = extract_embeddings(
        model, query_loader, device
    )

    print(f"  Gallery embeddings: {gallery_emb.shape}")
    print(f"  Query embeddings:   {query_emb.shape}")

    exp_name = cfg['train'].get('experiment_name', 'default')
    emb_path = f"runs/{exp_name}/embeddings.pt"
    os.makedirs(f"runs/{exp_name}", exist_ok=True)
    torch.save({
        'gallery_emb':    gallery_emb,
        'gallery_labels': gallery_labels,
        'query_emb':      query_emb,
        'query_labels':   query_labels
    }, emb_path)
    print(f"  Embeddings saved: {emb_path}")

    print("\n[4/4] Computing metrics...")

    rank_acc = compute_rank_metrics(
        query_emb, query_labels,
        gallery_emb, gallery_labels,
        ranks=(1, 5, 10)
    )

    map_score = compute_map(
        query_emb, query_labels,
        gallery_emb, gallery_labels
    )

    eer = compute_eer(
        query_emb, query_labels,
        gallery_emb, gallery_labels,
        max_query=1000
    )

    print(f"  Rank-1 accuracy: {rank_acc[1]:.2f}%")
    print(f"  Rank-5 accuracy: {rank_acc[5]:.2f}%")
    print(f"  Rank-10 accuracy:{rank_acc[10]:.2f}%")
    print(f"  mAP:             {map_score:.2f}%")
    if eer >= 0:
        print(f"  EER:             {eer:.2f}%")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return rank_acc[1], rank_acc[5], rank_acc[10], map_score, eer

if __name__ == '__main__':
    import yaml

    with open('configs/server.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg['train'].get('experiment_name', 'default')
    checkpoint_path = None

    for path_template in [
        f"runs/{exp_name}/best_checkpoint.pth",
        *[f"runs/{exp_name}/checkpoint_epoch_{e}.pth"
          for e in [60, 50, 40, 30, 20, 10, 5]]
    ]:
        if os.path.exists(path_template):
            checkpoint_path = path_template
            break

    print(f"Experiment:  {exp_name}")
    print(f"Checkpoint:  {checkpoint_path}")

    evaluate(
        cfg_path='configs/server.yaml',
        checkpoint_path=checkpoint_path
    )
