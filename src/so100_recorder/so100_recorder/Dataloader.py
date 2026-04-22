import os
import pandas as pd
import torch
import torchvision.io as io
from torch.utils.data import Dataset, DataLoader

class SO100Dataset(Dataset):
    def __init__(self, dataset_dir, chunk_size=16):
        """
        dataset_dir: Path to your 'so100_teleop' folder.
        chunk_size: How many future steps the U-Net needs to predict (default 16).
        """
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        
        # 1. Load the master episode list
        meta_path = os.path.join(dataset_dir, "meta", "episodes.parquet")
        self.episodes_df = pd.read_parquet(meta_path)
        
        # 2. Build a list of every single valid frame index across all episodes.
        self.valid_indices = []
        for index, row in self.episodes_df.iterrows():
            ep_id = row['episode_index']
            length = row['length']
            if length > self.chunk_size:
                for start_idx in range(length - self.chunk_size):
                    self.valid_indices.append((ep_id, start_idx))

        print(f"Loaded {len(self.episodes_df)} episodes. Total training samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_id, start_idx = self.valid_indices[idx]
        
        # Format the episode ID (e.g., 0 -> "000000")
        ep_str = f"episode_{ep_id:06d}"
        
        # Paths to the specific episode's data
        parquet_path = os.path.join(self.dataset_dir, "data", "chunk-000", f"{ep_str}.parquet")
        vid_dir = os.path.join(self.dataset_dir, "videos", "chunk-000") # <-- Fixed to match your LeRobot folder!
        
        # 1. Load the Parquet file for this episode
        ep_data = pd.read_parquet(parquet_path)
        
        # Slice the specific chunk of data we need (e.g., frames 40 to 56)
        chunk_data = ep_data.iloc[start_idx : start_idx + self.chunk_size]
        
        # Extract the actions and current state
        actions = torch.tensor(chunk_data['action'].tolist(), dtype=torch.float32)
        state = torch.tensor(chunk_data['observation.state'].tolist(), dtype=torch.float32)[0]
        
        # 2. Load the Images for the current frame
        overhead_vid_path = os.path.join(vid_dir, f"observation.images.overhead_{ep_str}.mp4")
        wrist_vid_path = os.path.join(vid_dir, f"observation.images.wrist_{ep_str}.mp4")
        
        overhead_frames, _, _ = io.read_video(overhead_vid_path, pts_unit='sec', output_format="TCHW")
        wrist_frames, _, _ = io.read_video(wrist_vid_path, pts_unit='sec', output_format="TCHW")
        
        overhead_img = overhead_frames[start_idx].float() / 255.0 # Normalize to [0, 1]
        wrist_img = wrist_frames[start_idx].float() / 255.0
        
        return {
            "overhead_img": overhead_img,
            "wrist_img": wrist_img,
            "state": state,
            "actions": actions
        }