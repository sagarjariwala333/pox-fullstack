import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

MODEL_PATH = r"d:\\pox\\models\\mpoxnet_v_fold2.pt"

class CrossAttentionGate(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, g: torch.Tensor, l: torch.Tensor):
        w = self.gate(torch.cat([g, l], dim=1))
        a, b = w[:, 0:1], w[:, 1:2]
        return a * g + b * l, a.squeeze(1), b.squeeze(1)


class MpoxNetV(nn.Module):
    FUSION_DIM = 1024

    def __init__(self, num_classes: int = 6, dropout: float = 0.35):
        super().__init__()
        self.num_classes = num_classes

        self.deit = timm.create_model(
            "deit_base_distilled_patch16_224", pretrained=False, num_classes=0
        )
        self.proj_g = nn.Sequential(
            nn.Linear(768 * 2, self.FUSION_DIM),
            nn.BatchNorm1d(self.FUSION_DIM),
            nn.GELU(),
        )

        self.effnet = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0
        )
        self.proj_l = nn.Sequential(
            nn.Linear(1792, self.FUSION_DIM),
            nn.BatchNorm1d(self.FUSION_DIM),
            nn.GELU(),
        )

        self.gate = CrossAttentionGate(dim=self.FUSION_DIM)

        self.head = nn.Sequential(
            nn.Linear(self.FUSION_DIM, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        tokens = self.deit.forward_features(x)
        G = self.proj_g(torch.cat([tokens[:, 0], tokens[:, 1]], dim=1))
        L = self.proj_l(self.effnet(x))
        F, alpha, beta = self.gate(G, L)
        logits = self.head(F)
        return (logits, alpha, beta) if return_gate else logits


checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model = MpoxNetV(num_classes=checkpoint.get('num_classes', 6))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the same transforms used during training (placeholder – adjust as needed)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels – order must match the model's output
# CLASS_LABELS = ["Monkeypox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
CLASS_LABELS = [
    "Chickenpox",
    "Measles",
    "Monkeypox",
    "Normal"
]

def predict(image_bytes: bytes) -> dict:
    """Run inference on an image and return label and probabilities.

    Args:
        image_bytes: Raw image data (e.g., from uploaded file).
    Returns:
        dict with keys: 'label' (str) and 'probabilities' (dict of class->prob).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # shape: [1, C, H, W]
    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).tolist()
    prob_dict = {cls: round(p, 4) for cls, p in zip(CLASS_LABELS, probs)}
    predicted_label = CLASS_LABELS[probs.index(max(probs))]
    return {"label": predicted_label, "probabilities": prob_dict}
