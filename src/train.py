from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import re
import os
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import av

from policy import ActionPolicy, POSITION_TO_ID


# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_ID = "ehalicki/eth-rl-so101-may4"
IMAGE_SIZE  = 256
FPS         = 30


# ─── Task index → bowl color ──────────────────────────────────────────────────

def build_task_index_to_bowl(repo_id: str = DATASET_ID) -> dict:
    """
    Loads meta/tasks.parquet and extracts bowl color from each task string via regex.
    Corrupted entries (arrow-key artifacts, typos) still contain a color word.
    Pure garbage (e.g. "[D11") returns None — those episodes are dropped later.
    """
    path = hf_hub_download(repo_id=repo_id, filename="meta/tasks.parquet", repo_type="dataset")
    tasks_df = pd.read_parquet(path)
    mapping = {}
    for task_str, row in tasks_df.iterrows():
        match = re.search(r"\b(green|blue|red)\b", task_str, re.IGNORECASE)
        mapping[int(row["task_index"])] = match.group(1).lower() if match else None
    return mapping


TASK_INDEX_TO_BOWL = build_task_index_to_bowl()

# Eval 1 fixed bowl order: blue=left, red=middle, green=right.
# Policy is conditioned on position so Eval 1 and Eval 2 share the same trained model.
# NOTE: verify your teleop episodes match this physical order before training.
BOWL_TO_POSITION = {
    "blue":  "left",
    "red":   "middle",
    "green": "right",
}


# ─── Episode → video file mapping ─────────────────────────────────────────────

def load_episode_video_map(repo_id: str = DATASET_ID) -> dict:
    """
    Returns {episode_index: {chunk_index, file_index, from_timestamp}} by reading
    all meta/episodes parquet files. Used to locate which video file contains each
    episode and at what timestamp it starts within that file.
    """
    seed_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/episodes/chunk-000/file-000.parquet",
        repo_type="dataset",
    )
    episodes_dir = os.path.dirname(os.path.dirname(seed_path))  # .../meta/episodes/

    episode_map = {}
    for path in sorted(glob.glob(os.path.join(episodes_dir, "**/*.parquet"), recursive=True)):
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            episode_map[int(row["episode_index"])] = {
                "chunk_index":    int(row["videos/observation.images.wrist_camera/chunk_index"]),
                "file_index":     int(row["videos/observation.images.wrist_camera/file_index"]),
                "from_timestamp": float(row["videos/observation.images.wrist_camera/from_timestamp"]),
            }
    return episode_map


def get_video_path(chunk_index: int, file_index: int, repo_id: str = DATASET_ID) -> str:
    filename = (
        f"videos/observation.images.wrist_camera/"
        f"chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class VLAPolicyDataset(Dataset):
    """
    Yields (image, state, position_id, action) tuples from the teleop dataset.

    - image:       Float tensor [3, 256, 256] in [0, 1]
    - state:       Float tensor [6]  — current joint positions
    - position_id: Long tensor scalar — 0=left, 1=middle, 2=right
    - action:      Float tensor [6]  — target joint positions
    """

    def __init__(self, df: pd.DataFrame, episode_video_map: dict, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.episode_video_map = episode_video_map
        self._video_path_cache: dict = {}

        self.resize = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
        ])
        self.augment = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2) if augment else None

    def _video_path(self, chunk_index: int, file_index: int) -> str:
        key = (chunk_index, file_index)
        if key not in self._video_path_cache:
            self._video_path_cache[key] = get_video_path(chunk_index, file_index)
        return self._video_path_cache[key]

    def _load_frame(self, episode_index: int, within_episode_ts: float) -> Image.Image:
        meta = self.episode_video_map[episode_index]
        video_ts = meta["from_timestamp"] + within_episode_ts  # absolute position in mp4
        video_path = self._video_path(meta["chunk_index"], meta["file_index"])

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            # Seek backward to nearest keyframe, then decode forward to target
            target_pts = int(video_ts / float(stream.time_base))
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if float(frame.pts * stream.time_base) >= video_ts - 1.0 / FPS:
                    return frame.to_image()

        raise RuntimeError(f"Frame not found at ts={video_ts:.3f}s in {video_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = self._load_frame(int(row["episode_index"]), float(row["timestamp"]))
        if self.augment:
            image = self.augment(image)
        image = self.resize(image)  # [3, 256, 256]

        state       = torch.tensor(list(row["observation.state"]), dtype=torch.float32)
        position_id = torch.tensor(POSITION_TO_ID[row["position"]], dtype=torch.long)
        action      = torch.tensor(list(row["action"]), dtype=torch.float32)

        return image, state, position_id, action


# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    checkpoint_path: str = "checkpoints/policy.pt",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    val_split: float = 0.1,
    num_workers: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Data ---
    print("Loading dataset...")
    ds = load_dataset(DATASET_ID, data_files={"train": "data/chunk-*/file-*.parquet"})
    df = pd.DataFrame(ds["train"])

    df["bowl"]     = df["task_index"].map(TASK_INDEX_TO_BOWL)
    df["position"] = df["bowl"].map(BOWL_TO_POSITION)
    df = df[df["bowl"].notna()].reset_index(drop=True)

    print(f"Frames after cleaning: {len(df)}")
    print(df["position"].value_counts())

    episode_video_map = load_episode_video_map()

    full_ds = VLAPolicyDataset(df, episode_video_map, augment=True)
    val_size   = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds.dataset.augment = None  # no augmentation on val

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # --- Model ---
    model = ActionPolicy(hidden_dim=256, freeze_encoder=True).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for images, states, position_ids, actions in train_loader:
            images, states      = images.to(device), states.to(device)
            position_ids        = position_ids.to(device)
            actions             = actions.to(device)

            preds = model(images, states, position_ids)
            loss  = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, states, position_ids, actions in val_loader:
                images, states  = images.to(device), states.to(device)
                position_ids    = position_ids.to(device)
                actions         = actions.to(device)
                preds = model(images, states, position_ids)
                val_loss += criterion(preds, actions).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(checkpoint_path)
            print(f"  → saved checkpoint (val_loss={val_loss:.5f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.5f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/policy.pt")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    args = parser.parse_args()

    train(
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
    )
