"""Combine two LeRobot datasets into a single new dataset.

Usage:
    python scripts/combine_datasets.py \
        --src ehalicki/eth-rl-so101-may4 \
        --src ehalicki/ethz-rl-so101-may17-eval1 \
        --dst ehalicki/eth-rl-so101-combined \
        [--push]
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset


NON_FRAME_KEYS = {
    "episode_index",
    "frame_index",
    "index",
    "timestamp",
    "task_index",
}


def resolve_task(src: LeRobotDataset, task_idx: int) -> str:
    tasks = src.meta.tasks
    matches = tasks.index[tasks["task_index"] == task_idx]
    return str(matches[0])


def copy_episodes(src: LeRobotDataset, dst: LeRobotDataset) -> int:
    hf = src.hf_dataset
    episode_col = hf["episode_index"]
    n_copied = 0

    current_ep = None
    pending = False
    for i in range(len(hf)):
        ep = int(episode_col[i])
        if current_ep is None:
            current_ep = ep
        elif ep != current_ep:
            dst.save_episode()
            n_copied += 1
            pending = False
            current_ep = ep

        row = hf[i]
        frame = {k: v for k, v in row.items() if k not in NON_FRAME_KEYS}
        task_idx = int(row["task_index"])
        frame["task"] = resolve_task(src, task_idx)
        for cam_key in src.meta.camera_keys:
            img = src[i][cam_key]
            if hasattr(img, "detach"):
                img = img.detach().cpu().numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = img.transpose(1, 2, 0)
            frame[cam_key] = img
        dst.add_frame(frame)
        pending = True

    if pending:
        dst.save_episode()
        n_copied += 1
    return n_copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="Source repo_id (repeat for multiple)")
    ap.add_argument("--dst", required=True, help="Destination repo_id")
    ap.add_argument("--root", default=None, help="Local root directory for the new dataset")
    ap.add_argument("--push", action="store_true", help="Push the combined dataset to the Hub")
    ap.add_argument("--private", action="store_true", help="If pushing, make repo private")
    ap.add_argument("--revision", default="main", help="Branch/revision to load from source repos")
    args = ap.parse_args()

    sources = [LeRobotDataset(repo_id, revision=args.revision) for repo_id in args.src]

    base = sources[0]
    fps = base.fps
    features = {k: v for k, v in base.features.items() if k not in NON_FRAME_KEYS}
    robot_type = base.meta.robot_type

    for s in sources[1:]:
        if s.fps != fps:
            raise ValueError(f"fps mismatch: {s.repo_id} has {s.fps}, expected {fps}")
        s_features = {k: v for k, v in s.features.items() if k not in NON_FRAME_KEYS}
        if set(s_features.keys()) != set(features.keys()):
            raise ValueError(
                f"feature keys mismatch between {sources[0].repo_id} and {s.repo_id}:\n"
                f"  {sorted(features.keys())}\n  vs\n  {sorted(s_features.keys())}"
            )

    dst = LeRobotDataset.create(
        repo_id=args.dst,
        fps=fps,
        features=features,
        root=args.root,
        robot_type=robot_type,
        use_videos=True,
    )

    total_eps = 0
    for s in sources:
        print(f"Copying {s.repo_id}: {s.num_episodes} episodes, {s.num_frames} frames")
        total_eps += copy_episodes(s, dst)

    dst.finalize()
    print(f"Done. Combined dataset has {total_eps} episodes at {dst.root}")

    if args.push:
        dst.push_to_hub(private=args.private)
        print(f"Pushed to {args.dst}")


if __name__ == "__main__":
    main()
