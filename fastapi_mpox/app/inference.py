import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

MODELS_DIR = r"d:\pox\models"
DEFAULT_MODEL = "mpoxnet_v_fold2.pt"

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


_model_cache = {}

def get_model(model_name: str):
    """Retrieve model from cache or load it from disk."""
    if model_name not in _model_cache:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            # Fallback for unexpected paths
            if not os.path.isabs(model_name):
                 model_path = os.path.join(MODELS_DIR, model_name)
            else:
                 model_path = model_name
                 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        m = MpoxNetV(num_classes=checkpoint.get('num_classes', 6))
        m.load_state_dict(checkpoint['model_state_dict'])
        m.eval()
        _model_cache[model_name] = m
    return _model_cache[model_name]

# Load default model initially
try:
    default_model = get_model(DEFAULT_MODEL)
except Exception as e:
    print(f"Error loading default model: {e}")
    default_model = None

# Define the same transforms used during training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class labels – order must match the model's output
CLASS_LABELS = [
    "Chickenpox",
    "Measles",
    "Monkeypox",
    "Normal"
]

def predict(image_bytes: bytes, model_name: str = None) -> dict:
    """Run inference on an image and return label and probabilities.

    Args:
        image_bytes: Raw image data (e.g., from uploaded file).
        model_name: Optional name of the model to use.
    Returns:
        dict with keys: 'label' (str) and 'probabilities' (dict of class->prob).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)  # shape: [1, C, H, W]
    
    current_model = get_model(model_name) if model_name else default_model
    if current_model is None:
        raise RuntimeError("No model available for prediction")
        
    with torch.no_grad():
        outputs = current_model(input_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).tolist()
    
    prob_dict = {cls: round(p, 4) for cls, p in zip(CLASS_LABELS, probs)}
    predicted_label = CLASS_LABELS[probs.index(max(probs))]
    return {"label": predicted_label, "probabilities": prob_dict}
