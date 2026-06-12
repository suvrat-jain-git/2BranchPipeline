import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.dataset import CCVIDDataset
from src.models.pipeline import Pipeline


def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for frames, label in loader:
            frames = frames.to(device)

            # forward pass — no labels in inference mode
            output = model(frames, labels=None)

            # l2 normalise identity embedding
            # [batch, 75] => [batch, 75] normalised
            emb = F.normalize(output['identity'], dim=1)

            embeddings.append(emb.cpu())
            labels.append(label)

    # stack all batches
    embeddings = torch.cat(embeddings, dim=0)  # [N, 75]
    labels = torch.cat(labels, dim=0)          # [N]

    return embeddings, labels


def compute_rank1_rank5(query_emb, query_labels,
                        gallery_emb, gallery_labels):
    # cosine similarity between all query and gallery embeddings
    # [num_query, num_gallery]
    sim_matrix = torch.mm(query_emb, gallery_emb.t())

    num_query = query_emb.shape[0]
    rank1_correct = 0
    rank5_correct = 0

    for i in range(num_query):
        # similarity scores for this query against all gallery
        sim = sim_matrix[i]           # [num_gallery]
        q_label = query_labels[i].item()

        # sort gallery by similarity descending
        sorted_indices = sim.argsort(descending=True)
        sorted_labels = gallery_labels[sorted_indices]

        # rank-1: is top-1 match correct?
        if sorted_labels[0].item() == q_label:
            rank1_correct += 1

        # rank-5: is correct match in top-5?
        if q_label in sorted_labels[:5].tolist():
            rank5_correct += 1

    rank1 = 100.0 * rank1_correct / num_query
    rank5 = 100.0 * rank5_correct / num_query
    return rank1, rank5


def compute_map(query_emb, query_labels,
                gallery_emb, gallery_labels):
    # cosine similarity matrix
    sim_matrix = torch.mm(query_emb, gallery_emb.t())

    num_query = query_emb.shape[0]
    average_precisions = []

    for i in range(num_query):
        sim = sim_matrix[i]
        q_label = query_labels[i].item()

        # sort gallery by similarity descending
        sorted_indices = sim.argsort(descending=True)
        sorted_labels = gallery_labels[sorted_indices]

        # find positions of correct matches
        correct_mask = (sorted_labels == q_label)
        num_correct = correct_mask.sum().item()

        if num_correct == 0:
            continue

        # compute average precision
        # precision at each correct match position
        positions = torch.where(correct_mask)[0].float() + 1  # 1-indexed
        precisions = torch.arange(1, num_correct + 1).float() / positions
        ap = precisions.mean().item()
        average_precisions.append(ap)

    map_score = 100.0 * sum(average_precisions) / len(average_precisions)
    return map_score


def compute_eer(query_emb, query_labels,
                gallery_emb, gallery_labels,
                max_query=50):
    num_query = query_emb.shape[0]

    # skip EER if too many pairs for CPU
    if num_query > max_query:
        print(f"  Skipping EER — {num_query} queries too many for CPU")
        print(f"  EER will be computed on server")
        return -1.0

    # compute all pairwise cosine similarities
    sim_matrix = torch.mm(query_emb, gallery_emb.t())

    scores = []
    is_genuine = []

    num_gallery = gallery_emb.shape[0]

    for i in range(num_query):
        for j in range(num_gallery):
            score = sim_matrix[i, j].item()
            genuine = (query_labels[i].item() == gallery_labels[j].item())
            scores.append(score)
            is_genuine.append(int(genuine))

    scores = torch.tensor(scores)
    is_genuine = torch.tensor(is_genuine)

    # sweep thresholds from -1 to 1
    thresholds = torch.linspace(-1, 1, steps=1000)
    min_diff = float('inf')
    eer = 0.0

    for threshold in thresholds:
        # predicted same person if score >= threshold
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


def evaluate(cfg_path='configs/smoke_test.yaml',
             checkpoint_path=None):
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'])
    print(f"\nDevice: {device}")

    # ── Model ─────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    model = Pipeline(cfg).to(device)

    # load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f"  Loaded checkpoint from epoch {epoch}")
    else:
        print("  No checkpoint provided — using random weights")
        print("  (for pipeline verification only)")

    # ── Datasets ──────────────────────────────────────────────────
    print("\n[2/4] Loading gallery and query datasets...")
    gallery_dataset = CCVIDDataset(cfg, split='gallery')
    query_dataset   = CCVIDDataset(cfg, split='query')

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers']
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers']
    )

    print(f"  Gallery sequences: {len(gallery_dataset)}")
    print(f"  Query sequences:   {len(query_dataset)}")

    # ── Extract embeddings ────────────────────────────────────────
    print("\n[3/4] Extracting embeddings...")
    gallery_emb, gallery_labels = extract_embeddings(
        model, gallery_loader, device
    )
    query_emb, query_labels = extract_embeddings(
        model, query_loader, device
    )

    print(f"  Gallery embeddings: {gallery_emb.shape}")
    print(f"  Query embeddings:   {query_emb.shape}")

    # ── Compute metrics ───────────────────────────────────────────
    print("\n[4/4] Computing metrics...")

    rank1, rank5 = compute_rank1_rank5(
        query_emb, query_labels,
        gallery_emb, gallery_labels
    )

    map_score = compute_map(
        query_emb, query_labels,
        gallery_emb, gallery_labels
    )

    eer = compute_eer(
        query_emb, query_labels,
        gallery_emb, gallery_labels,
        max_query=50
    )

    print(f"\n  Rank-1 accuracy: {rank1:.2f}%")
    print(f"  Rank-5 accuracy: {rank5:.2f}%")
    print(f"  mAP:             {map_score:.2f}%")
    if eer >= 0:
        print(f"  EER:             {eer:.2f}%")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return rank1, rank5, map_score, eer


if __name__ == '__main__':
    # use latest checkpoint if available
    checkpoint_path = None
    if os.path.exists('runs/checkpoint_epoch_6.pth'):
        checkpoint_path = 'runs/checkpoint_epoch_6.pth'

    evaluate(
        cfg_path='configs/smoke_test.yaml',
        checkpoint_path=checkpoint_path
    )