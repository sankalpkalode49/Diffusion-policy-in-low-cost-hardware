import os
import glob
import pandas as pd
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class SO100Dataset(Dataset):
    def __init__(self, dataset_dir, chunk_size=16):
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.valid_indices = []
        
        # 🔴 NEW: Action Normalization Bounds
        self.TICK_MIN = torch.tensor([0., 1698., 460., 3354.], dtype=torch.float32)
        self.TICK_MAX = torch.tensor([2500., 4090., 2677., 4095.], dtype=torch.float32)
        
        # --- 1. PARQUET LOADING ---
        data_dir = os.path.join(dataset_dir, "data")
        parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        
        if not parquet_files:
            raise FileNotFoundError(f"Could not find any .parquet files in {data_dir}.")
            
        print(f"🔍 Found {len(parquet_files)} raw data files. Merging...")
        parquet_files.sort()
        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            if 'episode_index' not in df.columns:
                ep_id = int(os.path.basename(file).split('_')[1].split('.')[0])
                df['episode_index'] = ep_id
            dfs.append(df)
            
        self.master_df = pd.concat(dfs, ignore_index=True)
        
        # --- 2. INDEX CALCULATION ---
        episodes = self.master_df.groupby('episode_index')
        for ep_id, group in episodes:
            length = len(group)
            if length > self.chunk_size:
                first_row = group.index[0] 
                for frame_idx in range(length - self.chunk_size):
                    self.valid_indices.append((ep_id, first_row + frame_idx))

        print(f"✅ Loaded {len(episodes)} episodes. Total training samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def _get_frame(self, absolute_row_idx):
        img_path = os.path.join(self.dataset_dir, "frames", f"frame_{absolute_row_idx:06d}.jpg")
        
        frame = cv2.imread(img_path)
        if frame is None:
            raise RuntimeError(f"Missing frame image: {img_path}")
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return frame_tensor

    def __getitem__(self, idx):
        ep_id, absolute_row_idx = self.valid_indices[idx]
        
        chunk_data = self.master_df.iloc[absolute_row_idx : absolute_row_idx + self.chunk_size]
        
        actions = torch.tensor(np.array(chunk_data['action'].tolist()), dtype=torch.float32)
        
        # 🔴 NEW: Scale actions to [-1, 1] range for DDPM
        actions = 2.0 * (actions - self.TICK_MIN) / (self.TICK_MAX - self.TICK_MIN) - 1.0
        
        state = torch.tensor(np.array(chunk_data['observation.state'].tolist()), dtype=torch.float32)[0]
        img = self._get_frame(absolute_row_idx)
        
        return {
            "img": img,
            "state": state,
            "actions": actions
        }