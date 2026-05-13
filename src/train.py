from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import os
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import av

from policy import VLAPolicy, TASK_INDEX_TO_STRING


# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_ID = "ehalicki/eth-rl-so101-may4"
IMAGE_SIZE  = 256
FPS         = 30


# ─── Episode → video file mapping ─────────────────────────────────────────────

def load_episode_video_map(repo_id: str = DATASET_ID) -> dict:
    """Returns {episode_index: {chunk_index, file_index, from_timestamp}}."""
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
    Yields (image, state, task_string, action) tuples.

    - image:       PIL Image (256×256) — VLAPolicy.encode_inputs handles normalisation
    - state:       Float tensor [6]
    - task_string: str — canonical task instruction
    - action:      Float tensor [6]
    """

    def __init__(self, df: pd.DataFrame, episode_video_map: dict, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.episode_video_map = episode_video_map
        self._video_path_cache: dict = {}

        self.resize  = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.augment = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2) if augment else None

    def _video_path(self, chunk_index: int, file_index: int) -> str:
        key = (chunk_index, file_index)
        if key not in self._video_path_cache:
            self._video_path_cache[key] = get_video_path(chunk_index, file_index)
        return self._video_path_cache[key]

    def _load_frame(self, episode_index: int, within_episode_ts: float) -> Image.Image:
        meta      = self.episode_video_map[episode_index]
        video_ts  = meta["from_timestamp"] + within_episode_ts
        video_path = self._video_path(meta["chunk_index"], meta["file_index"])

        with av.open(video_path) as container:
            stream     = container.streams.video[0]
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
        image = self.resize(image)
        if self.augment:
            image = self.augment(image)

        state       = torch.tensor(list(row["observation.state"]), dtype=torch.float32)
        task_string = row["task_string"]
        action      = torch.tensor(list(row["action"]), dtype=torch.float32)

        return image, state, task_string, action


def collate_fn(batch):
    """Keep PIL images and task strings as lists; stack tensors."""
    images, states, task_strings, actions = zip(*batch)
    return list(images), list(task_strings), torch.stack(states), torch.stack(actions)


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

    df["task_string"] = df["task_index"].map(TASK_INDEX_TO_STRING)
    df = df[df["task_string"].notna()].reset_index(drop=True)

    print(f"Frames after cleaning: {len(df)}")
    print(df["task_string"].value_counts())

    episode_video_map = load_episode_video_map()

    # Split indices to create separate dataset instances (avoids shared augment state)
    total      = len(df)
    val_size   = int(total * val_split)
    train_size = total - val_size
    perm       = torch.randperm(total).tolist()

    train_df = df.iloc[perm[:train_size]].reset_index(drop=True)
    val_df   = df.iloc[perm[train_size:]].reset_index(drop=True)

    train_ds = VLAPolicyDataset(train_df, episode_video_map, augment=True)
    val_ds   = VLAPolicyDataset(val_df,   episode_video_map, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=collate_fn)

    # --- Model ---
    model     = VLAPolicy(hidden_dim=256, freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for pil_images, task_strings, states, actions in train_loader:
            states  = states.to(device)
            actions = actions.to(device)

            inputs = model.encode_inputs(pil_images, task_strings, device)
            preds  = model(inputs.pixel_values, inputs.input_ids, inputs.attention_mask, states)
            loss   = criterion(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pil_images, task_strings, states, actions in val_loader:
                states  = states.to(device)
                actions = actions.to(device)
                inputs  = model.encode_inputs(pil_images, task_strings, device)
                preds   = model(inputs.pixel_values, inputs.input_ids, inputs.attention_mask, states)
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
