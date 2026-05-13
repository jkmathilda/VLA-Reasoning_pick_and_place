import sys
import types as _types

# Stub out the broken groot module before any lerobot.policies import
for _mod in ["lerobot.policies.groot", "lerobot.policies.groot.configuration_groot"]:
    if _mod not in sys.modules:
        _s = _types.ModuleType(_mod)
        _s.GrootConfig = None
        sys.modules[_mod] = _s

from datasets import load_dataset
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
import os
import glob
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import av
from transformers import AutoTokenizer

from lerobot.utils.constants import (
    OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION
)
from policy import VLAPolicy, build_task_index_to_string, IMG_KEY


# ─── Task paraphrases (Eval 1) ────────────────────────────────────────────────
# Maps each canonical task string to equivalent phrasings used during training.
# Keeps val/inference on canonical strings so metrics are comparable.

TASK_PARAPHRASES: dict[str, list[str]] = {
    "Put the banana in the blue bowl.": [
        "Put the banana in the blue bowl.",
        "Place the banana in the blue bowl.",
        "Put the banana into the blue bowl.",
        "Place the banana into the blue bowl.",
        "Move the banana to the blue bowl.",
    ],
    "Put the banana in the red bowl.": [
        "Put the banana in the red bowl.",
        "Place the banana in the red bowl.",
        "Put the banana into the red bowl.",
        "Place the banana into the red bowl.",
        "Move the banana to the red bowl.",
    ],
    "Put the banana in the green bowl.": [
        "Put the banana in the green bowl.",
        "Place the banana in the green bowl.",
        "Put the banana into the green bowl.",
        "Place the banana into the green bowl.",
        "Move the banana to the green bowl.",
    ],
}


# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_ID  = "ehalicki/eth-rl-so101-may4"
IMAGE_SIZE  = 256
FPS         = 30
CHUNK_SIZE  = 50   # must match SmolVLA config.chunk_size
TOKENIZER_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_MAX_LEN = 48


# ─── Episode → video file mapping ─────────────────────────────────────────────

def load_episode_video_map(repo_id: str = DATASET_ID) -> dict:
    """Returns {episode_index: {chunk_index, file_index, from_timestamp}}."""
    seed_path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/episodes/chunk-000/file-000.parquet",
        repo_type="dataset",
    )
    episodes_dir = os.path.dirname(os.path.dirname(seed_path))

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
    Yields (image, state, task_string, action_chunk, action_is_pad).

    - image:         PIL Image (256×256)
    - state:         Float tensor [6]
    - task_string:   str
    - action_chunk:  Float tensor [CHUNK_SIZE, 6]  — consecutive target actions
    - action_is_pad: Bool tensor  [CHUNK_SIZE]     — True for padded steps past episode end
    """

    def __init__(self, df: pd.DataFrame, episode_video_map: dict,
                 augment: bool = False, augment_tasks: bool = False,
                 chunk_size: int = CHUNK_SIZE):
        # Sort by episode then time so consecutive rows are the next steps
        self.df = df.sort_values(["episode_index", "timestamp"]).reset_index(drop=True)
        self.episode_video_map = episode_video_map
        self.chunk_size = chunk_size
        self._video_path_cache: dict = {}

        self.augment_tasks = augment_tasks
        self.resize  = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.augment = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2) if augment else None

        # Pre-compute chunk end indices per sample
        self._chunk_ends = self._build_chunk_ends()

    def _build_chunk_ends(self) -> list:
        """For each row i, store the last valid index in its episode chunk."""
        ep_col = self.df["episode_index"].to_numpy()
        ends = []
        n = len(self.df)
        for i in range(n):
            j = i
            while j < n - 1 and ep_col[j + 1] == ep_col[i] and (j - i) < self.chunk_size - 1:
                j += 1
            ends.append(j)
        return ends

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
        if self.augment_tasks:
            task_string = random.choice(TASK_PARAPHRASES.get(task_string, [task_string]))

        # Build action chunk [chunk_size, 6]; pad past episode end with last action
        end_idx  = self._chunk_ends[idx]
        n_real   = end_idx - idx + 1
        actions  = [
            torch.tensor(list(self.df.iloc[idx + k]["action"]), dtype=torch.float32)
            if k < n_real
            else torch.tensor(list(self.df.iloc[end_idx]["action"]), dtype=torch.float32)
            for k in range(self.chunk_size)
        ]
        action_chunk  = torch.stack(actions)                                    # [chunk_size, 6]
        action_is_pad = torch.tensor([k >= n_real for k in range(self.chunk_size)])  # [chunk_size]

        return image, state, task_string, action_chunk, action_is_pad


# ─── Collate & normalisation ──────────────────────────────────────────────────

def make_collate_fn(tokenizer, dataset_stats: dict, device: torch.device):
    state_mean  = dataset_stats["state"]["mean"].to(device)
    state_std   = dataset_stats["state"]["std"].to(device)
    action_mean = dataset_stats["action"]["mean"].to(device)
    action_std  = dataset_stats["action"]["std"].to(device)

    def collate_fn(batch):
        images, states, task_strings, action_chunks, is_pads = zip(*batch)

        # PIL → float tensor [B, 3, H, W] in [0, 1]
        to_tensor = T.ToTensor()
        img_batch = torch.stack([to_tensor(img) for img in images])

        states_t  = torch.stack(states)
        actions_t = torch.stack(action_chunks)   # [B, chunk_size, 6]
        is_pad_t  = torch.stack(is_pads)

        # Tokenise task strings (SmolVLA expects trailing newline)
        enc = tokenizer(
            [t + "\n" for t in task_strings],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=TOKENIZER_MAX_LEN,
        )

        # Move to device and normalise
        img_batch = img_batch.to(device)
        states_n  = ((states_t.to(device) - state_mean) / state_std)
        actions_n = ((actions_t.to(device) - action_mean) / action_std)

        return {
            IMG_KEY:                    img_batch,
            OBS_STATE:                  states_n,
            OBS_LANGUAGE_TOKENS:        enc["input_ids"].to(device),
            OBS_LANGUAGE_ATTENTION_MASK: enc["attention_mask"].to(device),
            ACTION:                     actions_n,
            "action_is_pad":            is_pad_t.to(device),
        }

    return collate_fn


def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute mean/std for state and action from the full training split."""
    states  = np.array([list(v) for v in df["observation.state"]], dtype=np.float32)
    actions = np.array([list(v) for v in df["action"]], dtype=np.float32)
    return {
        "state":  {"mean": torch.tensor(states.mean(0)),  "std": torch.tensor(states.std(0) + 1e-6)},
        "action": {"mean": torch.tensor(actions.mean(0)), "std": torch.tensor(actions.std(0) + 1e-6)},
    }


# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    checkpoint_path: str = "checkpoints/smolvla",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_split: float = 0.1,
    num_workers: int = 0,   # 0 = main process (avoids lerobot import issues in workers)
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- Data ---
    print("Loading dataset...")
    ds = load_dataset(DATASET_ID, data_files={"train": "data/chunk-*/file-*.parquet"})
    df = pd.DataFrame(ds["train"])

    task_map = build_task_index_to_string(DATASET_ID)
    df["task_string"] = df["task_index"].map(task_map)
    df = df[df["task_string"].notna()].reset_index(drop=True)

    print(f"Frames after cleaning: {len(df)}")
    print(df["task_string"].value_counts())

    episode_video_map = load_episode_video_map()

    total    = len(df)
    val_size = int(total * val_split)
    perm     = torch.randperm(total).tolist()

    train_df = df.iloc[perm[:total - val_size]].reset_index(drop=True)
    val_df   = df.iloc[perm[total - val_size:]].reset_index(drop=True)

    dataset_stats = compute_dataset_stats(train_df)

    train_ds = VLAPolicyDataset(train_df, episode_video_map, augment=True,  augment_tasks=True)
    val_ds   = VLAPolicyDataset(val_df,   episode_video_map, augment=False, augment_tasks=False)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.padding_side = "right"

    collate = make_collate_fn(tokenizer, dataset_stats, device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate)

    # --- Model ---
    policy    = VLAPolicy(dataset_stats=dataset_stats, device=device)
    smolvla   = policy.model   # the underlying SmolVLAPolicy nn.Module
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, smolvla.parameters()),
        lr=lr, betas=(0.9, 0.95), weight_decay=1e-10,
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        smolvla.train()
        train_loss = 0.0
        for batch in train_loader:
            loss, _ = smolvla.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(smolvla.parameters(), 10.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        smolvla.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = smolvla.forward(batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            policy.save(checkpoint_path)
            print(f"  → saved checkpoint (val={val_loss:.5f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.5f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/smolvla")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=0)
    args = parser.parse_args()

    train(
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
    )
