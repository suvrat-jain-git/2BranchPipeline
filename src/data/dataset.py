import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CCVIDDataset(Dataset):
    # cfg: config dict with keys
    def __init__(self, cfg, split='train'):
        self.root = cfg['dataset']['root']
        self.num_frames = cfg['dataset']['num_frames']
        self.sampling = cfg['dataset']['sampling']
        self.image_size = cfg['dataset']['image_size']
        self.crop_size = cfg['dataset']['crop_size']
        
        # select correct txt file based on split
        if split == 'train':
            txt_path = cfg['dataset']['train_txt']
        elif split == 'query':
            txt_path = os.path.join(self.root, 'query.txt')
        elif split == 'gallery':
            txt_path = os.path.join(self.root, 'gallery.txt')
        else:
            raise ValueError(f"Unknown split: {split}")

        # parse txt file
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

        # build identity string to integer index mapping
        unique_ids = sorted(set(identity_set))
        self.identity_to_idx = {pid: idx for idx, pid in enumerate(unique_ids)}

        # define transforms
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
            # if sequence shorter than num_frames, repeat last frame
            indices = list(range(total))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
        else:
            if self.sampling == 'uniform':
                # evenly spaced indices across the sequence
                indices = np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()
            elif self.sampling == 'random':
                # random contiguous clip of num_frames
                start = np.random.randint(0, total - self.num_frames)
                indices = list(range(start, start + self.num_frames))
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling}")

        return [frame_paths[i] for i in indices]

    def __getitem__(self, idx):
        seq_path, identity = self.sequences[idx]

        # full path to sequence folder
        full_seq_path = os.path.join(self.root, seq_path)

        # get all jpg frames sorted by filename
        frame_files = sorted([
            f for f in os.listdir(full_seq_path)
            if f.endswith('.jpg')
        ])

        # build full paths
        frame_paths = [
            os.path.join(full_seq_path, f) for f in frame_files
        ]

        # sample T frames
        sampled_paths = self._sample_frames(frame_paths)

        # load, transform and stack frames
        frames = []
        for path in sampled_paths:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            frames.append(img)

        # stack: list of T tensors (3, H, W) => (T, 3, H, W)
        frames = torch.stack(frames, dim=0)

        # integer label
        label = self.identity_to_idx[identity]

        return frames, label 

if __name__ == '__main__':
    import yaml

    with open('configs/smoke_test.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    dataset = CCVIDDataset(cfg, split='train')

    print(f"Total sequences: {len(dataset)}")
    print(f"Number of unique identities: {len(dataset.identity_to_idx)}")

    frames, label = dataset[0]
    print(f"Frames tensor shape: {frames.shape}")
    print(f"Label: {label}")
    print(f"Frame dtype: {frames.dtype}")
    print(f"Frame min: {frames.min():.3f}, max: {frames.max():.3f}")