"""Non-YOLO sequence models: the network definitions export_onnx.py needs to load their weights.

``shoplifting``: InsightEye's shoplifting classifier (Conv1d over time -> BiGRU ->
self-attention -> mean pool -> 2 logits) over a window of one person's pose
features [120 samples, 17 COCO keypoints, 12 features]. The network is copied
unchanged from InsightEye (via /home/ubuntu/shoplifting backend
app/services/shoplifting_model.py); the weights are a bare state_dict.

The exported graph is ``ExportWrapper``: z-score over the whole window (as in
training, population std like numpy's ``.std()``) -> network -> softmax, so the
engine takes raw features and returns class probabilities. Feature extraction
(numpy) lives in shared/yolo_platform_shared/behavior.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_FRAMES = 120
NUM_KEYPOINTS = 17
NUM_FEATURES = 12
SHOPLIFTING_NAMES = {0: "Normal", 1: "Shoplifting"}


class ShopliftingClassifier(nn.Module):
    def __init__(self, T: int = NUM_FRAMES, V: int = NUM_KEYPOINTS, C: int = NUM_FEATURES, num_classes: int = 2):
        super().__init__()
        self.T, self.V, self.C = T, V, C
        self.conv1 = nn.Conv1d(V * C, 128, kernel_size=3, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.gru = nn.GRU(128, 128, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.layernorm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.reshape(B, self.T, self.V * self.C).transpose(1, 2)   # (B, V*C, T)
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = x.transpose(1, 2)                                       # (B, T, 128)
        x, _ = self.gru(x)                                          # (B, T, 256)
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.layernorm(x + attn_out)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ExportWrapper(nn.Module):
    """raw features [B,T,V,C] -> per-window z-score -> network -> softmax [B, classes]."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        flat = sequences.reshape(sequences.size(0), -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = torch.sqrt(((flat - mean) ** 2).mean(dim=1, keepdim=True))
        x = ((flat - mean) / (std + 1e-7)).reshape(sequences.shape)
        return torch.softmax(self.net(x), dim=1)


def load_shoplifting(path: str) -> ShopliftingClassifier:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    net = ShopliftingClassifier()
    net.load_state_dict(state)
    return net.eval()


# builder name (models.yaml ``builder``) -> (loader, input shape without batch, class names, architecture)
BUILDERS = {
    "shoplifting": (load_shoplifting, [NUM_FRAMES, NUM_KEYPOINTS, NUM_FEATURES], SHOPLIFTING_NAMES,
                    "ShopliftingClassifier"),
}
