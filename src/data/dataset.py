import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CCVIDDataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.root = cfg['dataset']['root']
        self.num_frames = cfg['dataset']['num_frames']
        self.sampling = cfg['dataset']['sampling']
        self.image_size = cfg['dataset']['image_size']
        self.crop_size = cfg['dataset']['crop_size']
        
        if split == 'train':
            txt_path = cfg['dataset']['train_txt']
        elif split == 'query':
            txt_path = os.path.join(self.root, 'query.txt')
        elif split == 'gallery':
            txt_path = os.path.join(self.root, 'gallery.txt')
        else:
            raise ValueError(f"Unknown split: {split}")

        self.sequences = []
        identity_set = []

        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip().replace('\r', '')
                if not line:
                    continue
                parts = line.split()
                seq_path = parts[0]        
                identity = parts[1]
                self.sequences.append((seq_path, identity))
                identity_set.append(identity)

        unique_ids = sorted(set(identity_set))
        self.identity_to_idx = {pid: idx for idx, pid in enumerate(unique_ids)}

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.crop_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.sequences)

    def _sample_frames(self, frame_paths):
        total = len(frame_paths)

        if total == 0:
            raise RuntimeError(f"Empty sequence: {frame_paths}")

        if total <= self.num_frames:
            indices = list(range(total))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        else:
            if self.sampling == 'uniform':
                indices = np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()
            elif self.sampling == 'random':
                start = np.random.randint(0, total - self.num_frames + 1)
                indices = list(range(start, start + self.num_frames))
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling}")

        return [frame_paths[i] for i in indices]

    def __getitem__(self, idx):
        seq_path, identity = self.sequences[idx]

        full_seq_path = os.path.join(self.root, seq_path)
        
        frame_files = sorted([
            f for f in os.listdir(full_seq_path)
            if f.endswith('.jpg')
        ])

        frame_paths = [
            os.path.join(full_seq_path, f) for f in frame_files
        ]

        sampled_paths = self._sample_frames(frame_paths)

        frames = []
        for path in sampled_paths:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            frames.append(img)

        # T tensors (3, H, W) => (T, 3, H, W)
        frames = torch.stack(frames, dim=0)

        # integer label
        label = self.identity_to_idx[identity]

        return frames, label 
