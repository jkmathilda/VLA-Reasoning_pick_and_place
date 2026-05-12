import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

POSITION_TO_ID = {"left": 0, "middle": 1, "right": 2}
ID_TO_POSITION = {v: k for k, v in POSITION_TO_ID.items()}


class ActionPolicy(nn.Module):
    """
    Behavior cloning policy: (image, state, position_id) → 6-DOF action.

    Image encoder: pretrained ResNet-18 (frozen by default), outputs 512-dim feature.
    Position encoding: learned embedding for {left=0, middle=1, right=2}.
    MLP head: (512 + 6 + 16) → 256 → 256 → 6.
    """

    def __init__(self, hidden_dim: int = 256, freeze_encoder: bool = True):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # [B, 512, 1, 1]
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.position_emb = nn.Embedding(3, 16)
        self.mlp = nn.Sequential(
            nn.Linear(512 + 6 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(
        self,
        image: torch.Tensor,       # [B, 3, 256, 256]
        state: torch.Tensor,       # [B, 6]
        position_id: torch.Tensor, # [B]
    ) -> torch.Tensor:             # [B, 6]
        feat = self.encoder(image).squeeze(-1).squeeze(-1)  # [B, 512]
        pos_emb = self.position_emb(position_id)             # [B, 16]
        x = torch.cat([feat, state, pos_emb], dim=-1)
        return self.mlp(x)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "ActionPolicy":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
