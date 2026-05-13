import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# task_index → canonical task string (corrupted entries normalised)
TASK_INDEX_TO_STRING = {
    0: "Put the banana in the green bowl.",
    1: "Put the banana in the blue bowl.",
    2: "Put the banana in the red bowl.",
    3: "Put the banana in the red bowl.",   # corrupted typo → same intent
    4: "Put the banana in the blue bowl.",  # corrupted [D → same intent
    5: "Put the banana in the green bowl.", # corrupted [D → same intent
    6: None,                                # pure garbage → dropped
    7: "Put the banana in the blue bowl.",  # corrupted [C → same intent
    8: "Put the banana in the green bowl.", # alternate spelling → same intent
}

# Robot hardware constants (match record.py / teleop.py)
ROBOT_PORT  = "/dev/tty.usbmodem5B141136531"
CAMERA_DEFAULT_W, CAMERA_DEFAULT_H = 1920, 1080
CAMERA_CAP_W, CAMERA_CAP_H = 640, 480


class VLAPolicy(nn.Module):
    """
    Single VLA model: (image, task_string, state) → 6-DOF action.

    SmolVLM2 backbone (frozen by default) provides visual + language features.
    A lightweight MLP action head maps the backbone output + joint state to actions.
    """

    MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def __init__(self, hidden_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.backbone  = AutoModelForImageTextToText.from_pretrained(self.MODEL_ID)
        self._freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        backbone_dim = self.backbone.config.text_config.hidden_size
        self.action_head = nn.Sequential(
            nn.Linear(backbone_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_backbone:
            self.backbone.eval()  # keep frozen backbone in eval mode during training
        return self

    def encode_inputs(
        self,
        pil_images: list,
        task_strings: list,
        device: torch.device,
    ):
        """Run SmolVLM2 processor on a batch of PIL images + task strings."""
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": task},
                ]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for task in task_strings
        ]
        return self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        ).to(device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Returns predicted 6-DOF action tensor [B, 6]."""
        out = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden   = out.hidden_states[-1]                                    # [B, seq_len, D]
        seq_lens = attention_mask.sum(dim=1) - 1                           # last real token idx
        feat     = hidden[torch.arange(hidden.size(0), device=hidden.device), seq_lens]  # [B, D]
        return self.action_head(torch.cat([feat, state], dim=-1))

    def run(self, task: str, timeout: float = 18.0):
        """Inference loop: capture wrist-camera frames → predict → send to SO101."""
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig

        device = next(self.action_head.parameters()).device
        self.eval()

        # Discover wrist camera by its default resolution
        camera_idx = None
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                break
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == CAMERA_DEFAULT_W and h == CAMERA_DEFAULT_H:
                camera_idx = i
                break
        if camera_idx is None:
            raise RuntimeError(f"Wrist camera ({CAMERA_DEFAULT_W}×{CAMERA_DEFAULT_H}) not found")

        robot = SO101Follower(SO101FollowerConfig(
            port=ROBOT_PORT,
            id="eth_rl_so101_follower",
            cameras={"wrist_camera": OpenCVCameraConfig(
                index_or_path=camera_idx,
                width=CAMERA_CAP_W, height=CAMERA_CAP_H, fps=30,
            )},
        ))
        robot.connect()

        start = time.time()
        try:
            while time.time() - start < timeout:
                obs      = robot.capture_observation()
                frame_np = obs["observation.images.wrist_camera"]
                state_np = obs["observation.state"]

                frame_pil = Image.fromarray(frame_np) if isinstance(frame_np, np.ndarray) else frame_np
                state     = torch.tensor(list(state_np), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    inputs = self.encode_inputs([frame_pil], [task], device)
                    action = self(inputs.pixel_values, inputs.input_ids, inputs.attention_mask, state)

                robot.send_action({"action": action.squeeze(0).cpu().numpy()})
        finally:
            robot.disconnect()

    def save(self, path: str):
        """Save only the action head weights (backbone stays pretrained)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.action_head.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "VLAPolicy":
        model = cls(**kwargs)
        model.action_head.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        return model
