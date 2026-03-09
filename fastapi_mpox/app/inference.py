import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

MODEL_PATHS = [
    r"d:\pox\models\mpoxnet_v4class_fixed.pt",
    r"d:\pox\models\mpoxnet_v_fold2.pt",
]

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

    def __init__(self, num_classes: int = 4, dropout: float = 0.35):
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


# ─── Load Models (Ensemble) ───────────────────────────────────────────────────

models = []
for path in MODEL_PATHS:
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    num_classes = checkpoint.get('num_classes', 4)
    model = MpoxNetV(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append((model, num_classes))
    print(f"✅ Loaded: {path} | num_classes={num_classes}")

print(f"✅ Ensemble of {len(models)} models ready")

# ─── Class Labels (MUST match training order) ─────────────────────────────────

# 4-class labels (used by both v4class_fixed and fold2)
CLASS_LABELS = [
    "Chickenpox",
    "Measles",
    "Monkeypox",
    "Normal"
]

print(f"✅ Using labels: {CLASS_LABELS}")

# ─── Preprocessing ────────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Confidence Threshold ─────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.60  # below this → "Uncertain"

# Classes that are clinically relevant to show directly
PRIMARY_CLASSES = {"Monkeypox", "Chickenpox", "Measles"}

# ─── Predict Function ─────────────────────────────────────────────────────────

def predict(image_bytes: bytes) -> dict:
    """
    Run ensemble inference on an image and return label and probabilities.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # [1, C, H, W]

    # Average probabilities across all models
    avg_probs = torch.zeros(1, 4)  # 4-class: Chickenpox, Measles, Monkeypox, Normal

    with torch.no_grad():
        for model, nc in models:
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.nn.functional.softmax(outputs[:, :4], dim=1)
            avg_probs += probs

    avg_probs /= len(models)
    probs = avg_probs.squeeze(0).tolist()

    # Map all class probabilities
    prob_dict = {cls: round(p, 4) for cls, p in zip(CLASS_LABELS, probs)}

    # Top prediction
    top_idx = probs.index(max(probs))
    raw_label = CLASS_LABELS[top_idx]
    confidence = round(probs[top_idx], 4)

    # Uncertain if below threshold
    if confidence < CONFIDENCE_THRESHOLD:
        display_label = "Uncertain - requires clinical review"
        uncertain = True
    else:
        display_label = raw_label if raw_label in PRIMARY_CLASSES else "Other"
        uncertain = False

    return {
        "label": display_label,
        "raw_label": raw_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "uncertain": uncertain,
    }


# ─── Quick Sanity Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    with open(image_path, "rb") as f:
        result = predict(f.read())

    print("\n─── Prediction ───────────────────────────")
    print(f"  Label      : {result['label']}")
    print(f"  Raw Label  : {result['raw_label']}")
    print(f"  Confidence : {result['confidence'] * 100:.1f}%")
    print(f"  Uncertain  : {result['uncertain']}")
    print("\n─── All Probabilities ────────────────────")
    for cls, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob * 100:5.1f}%  {bar}")
    print("──────────────────────────────────────────")