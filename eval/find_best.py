import torch
import timm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import io
from collections import Counter

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — Update paths only
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = r"d:\pox\models"   # folder containing all .pt files
TEST_DIR  = r"d:\pox\dataset-1\MonkeyPox Skin Image Dataset for Computer Vision (\MPox-Vision"     # your 600 image test set

MODEL_PATHS = {
    # "fold1"          : os.path.join(MODEL_DIR, "mpoxnet_v_fold1.pt"),
    "fold2"          : os.path.join(MODEL_DIR, "mpoxnet_v_fold2.pt"),
    # "fold3"          : os.path.join(MODEL_DIR, "mpoxnet_v_fold3.pt"),
    # "fold3_1"        : os.path.join(MODEL_DIR, "mpoxnet_v_fold3-1.pt"),
    # "fold4"          : os.path.join(MODEL_DIR, "mpoxnet_v_fold4.pt"),
    "v4class_fixed"  : os.path.join(MODEL_DIR, "mpoxnet_v4class_fixed.pt"),
    # "fixed_2"        : os.path.join(MODEL_DIR, "mpoxnet_v4class_fixed_fixed_2.pt"),
    # "fixed_3"        : os.path.join(MODEL_DIR, "mpoxnet_v4class_fixed_fixed_3.pt"),
}

CLASS_LABELS = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionGate(nn.Module):
    def __init__(self, dim=1024, hidden=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(hidden, 2), nn.Softmax(dim=1),
        )
    def forward(self, g, l):
        w = self.gate(torch.cat([g, l], dim=1))
        a, b = w[:, 0:1], w[:, 1:2]
        return a * g + b * l, a.squeeze(1), b.squeeze(1)


class MpoxNetV(nn.Module):
    FUSION_DIM = 1024
    def __init__(self, num_classes=4, dropout=0.35):
        super().__init__()
        self.num_classes = num_classes
        self.deit   = timm.create_model("deit_base_distilled_patch16_224", pretrained=False, num_classes=0)
        self.proj_g = nn.Sequential(nn.Linear(768*2, self.FUSION_DIM), nn.BatchNorm1d(self.FUSION_DIM), nn.GELU())
        self.effnet = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0)
        self.proj_l = nn.Sequential(nn.Linear(1792, self.FUSION_DIM), nn.BatchNorm1d(self.FUSION_DIM), nn.GELU())
        self.gate   = CrossAttentionGate(dim=self.FUSION_DIM)
        self.head   = nn.Sequential(nn.Linear(self.FUSION_DIM, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, num_classes))
    def forward(self, x, return_gate=False):
        tokens = self.deit.forward_features(x)
        G = self.proj_g(torch.cat([tokens[:, 0], tokens[:, 1]], dim=1))
        L = self.proj_l(self.effnet(x))
        F, alpha, beta = self.gate(G, L)
        logits = self.head(F)
        return (logits, alpha, beta) if return_gate else logits


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATASET
# ══════════════════════════════════════════════════════════════════════════════

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CAP_PER_CLASS = 100   # ← change to 200 for full evaluation later

class TestDataset(Dataset):
    def __init__(self, root, cap=None):
        self.samples = []
        import random
        for label_idx, cls in enumerate(CLASS_LABELS):
            cls_dir = os.path.join(root, cls)
            if not os.path.exists(cls_dir):
                print(f"⚠️  Missing test folder: {cls_dir}")
                continue
            files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
            ]
            if cap:
                random.seed(42)
                files = random.sample(files, min(cap, len(files)))
            for f in files:
                self.samples.append((os.path.join(cls_dir, f), label_idx))
        print(f"✅ Test set: {len(self.samples)} images (capped at {cap} per class)")
        counts = Counter([s[1] for s in self.samples])
        for idx, cls in enumerate(CLASS_LABELS):
            print(f"   {cls:<15}: {counts.get(idx, 0)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return transform(Image.open(path).convert("RGB")), label, path


loader = DataLoader(TestDataset(TEST_DIR, cap=CAP_PER_CLASS),
                    batch_size=32, shuffle=False,
                    num_workers=0, pin_memory=False)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE SINGLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, loader, num_classes=4):
    model.eval()
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes
    chkpox_as_mpox = 0   # track the specific confusion
    mpox_as_chkpox = 0

    CHKPOX_IDX = CLASS_LABELS.index("Chickenpox")
    MPOX_IDX   = CLASS_LABELS.index("Monkeypox")

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]
            preds = outputs.argmax(dim=1)

            for p, l in zip(preds, labels):
                p, l = p.item(), l.item()
                if l < num_classes:
                    class_total[l]   += 1
                    if p == l:
                        class_correct[l] += 1
                if l == CHKPOX_IDX and p == MPOX_IDX:
                    chkpox_as_mpox += 1
                if l == MPOX_IDX and p == CHKPOX_IDX:
                    mpox_as_chkpox += 1

    overall    = sum(class_correct) / max(sum(class_total), 1) * 100
    per_class  = [class_correct[i] / max(class_total[i], 1) * 100 for i in range(num_classes)]

    return {
        "overall"       : overall,
        "per_class"     : per_class,
        "class_correct" : class_correct,
        "class_total"   : class_total,
        "chkpox_as_mpox": chkpox_as_mpox,   # Chickenpox wrongly called Monkeypox
        "mpox_as_chkpox": mpox_as_chkpox,   # Monkeypox wrongly called Chickenpox
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ALL 8 MODELS
# ══════════════════════════════════════════════════════════════════════════════

all_results  = {}
all_probs    = {}   # store raw probabilities for ensemble later

print("\n" + "═"*95)
print(f"{'Model':<18} {'Overall':>8} {'Chickenpox':>12} {'Measles':>10} {'Monkeypox':>12} {'Normal':>8} {'C→M':>6} {'M→C':>6}")
print("─"*95)

for name, path in MODEL_PATHS.items():
    if not os.path.exists(path):
        print(f"{name:<18} ⚠️  File not found: {path}")
        continue

    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    nc    = ckpt.get('num_classes', 4)
    model = MpoxNetV(num_classes=nc).to(DEVICE)

    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    except RuntimeError:
        # Handle size mismatch gracefully
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    model.eval()
    r = evaluate_model(model, loader, num_classes=min(nc, 4))
    all_results[name] = r

    pc = r["per_class"]
    pc_padded = pc + [0.0] * (4 - len(pc))  # pad with 0 if fewer classes
    print(
        f"{name:<18} "
        f"{r['overall']:>7.1f}% "
        f"{pc_padded[0]:>11.1f}% "
        f"{pc_padded[1]:>9.1f}% "
        f"{pc_padded[2]:>11.1f}% "
        f"{pc_padded[3]:>7.1f}% "
        f"{r['chkpox_as_mpox']:>6} "
        f"{r['mpox_as_chkpox']:>6}"
    )

print("═"*95)
print("C→M = Chickenpox predicted as Monkeypox (should be 0)")
print("M→C = Monkeypox predicted as Chickenpox (should be 0)")


# ══════════════════════════════════════════════════════════════════════════════
# BEST MODEL PER CLASS
# ══════════════════════════════════════════════════════════════════════════════

print("\n🏆 Best model per class:")
for i, cls in enumerate(CLASS_LABELS):
    valid = [(name, r) for name, r in all_results.items() if len(r["per_class"]) > i]
    if not valid: continue
    best_model = max(valid, key=lambda x: x[1]["per_class"][i])
    print(f"   {cls:<15}: {best_model[0]:<20} {best_model[1]['per_class'][i]:.1f}%")

best_overall = max(all_results.items(), key=lambda x: x[1]["overall"])
print(f"\n   {'Overall':<15}: {best_overall[0]:<20} {best_overall[1]['overall']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE AVERAGE ENSEMBLE (all 8 models)
# ══════════════════════════════════════════════════════════════════════════════

print("\n🔄 Computing ensemble predictions...")

# Collect all model outputs
loaded_models = {}
for name, path in MODEL_PATHS.items():
    if not os.path.exists(path): continue
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    nc    = ckpt.get('num_classes', 4)
    model = MpoxNetV(num_classes=nc).to(DEVICE)
    try:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    except RuntimeError:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    loaded_models[name] = model

# Run ensemble evaluation
CHKPOX_IDX = CLASS_LABELS.index("Chickenpox")
MPOX_IDX   = CLASS_LABELS.index("Monkeypox")

ens_correct  = [0] * 4
ens_total    = [0] * 4
ens_c2m      = 0
ens_m2c      = 0

with torch.no_grad():
    for images, labels, _ in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Average probabilities across all models
        avg_probs = torch.zeros(images.size(0), 4).to(DEVICE)
        for name, model in loaded_models.items():
            out = model(images)
            if isinstance(out, tuple): out = out[0]
            # Handle 6-class models — take only first 4 outputs
            out = out[:, :4]
            avg_probs += torch.softmax(out, dim=1)

        avg_probs /= len(loaded_models)
        preds      = avg_probs.argmax(dim=1)

        for p, l in zip(preds, labels):
            p, l = p.item(), l.item()
            if l < 4:
                ens_total[l]   += 1
                if p == l:
                    ens_correct[l] += 1
            if l == CHKPOX_IDX and p == MPOX_IDX: ens_c2m += 1
            if l == MPOX_IDX and p == CHKPOX_IDX: ens_m2c += 1

ens_overall   = sum(ens_correct) / max(sum(ens_total), 1) * 100
ens_per_class = [ens_correct[i] / max(ens_total[i], 1) * 100 for i in range(4)]

print("\n" + "═"*95)
print(f"{'ENSEMBLE (8 models)':<18} "
      f"{ens_overall:>7.1f}% "
      f"{ens_per_class[0]:>11.1f}% "
      f"{ens_per_class[1]:>9.1f}% "
      f"{ens_per_class[2]:>11.1f}% "
      f"{ens_per_class[3]:>7.1f}% "
      f"{ens_c2m:>6} "
      f"{ens_m2c:>6}")
print("═"*95)

print("\n📊 Summary:")
print(f"   Best single model : {best_overall[0]} ({best_overall[1]['overall']:.1f}%)")
print(f"   8-model ensemble  : {ens_overall:.1f}%")
print(f"   Gain from ensemble: +{ens_overall - best_overall[1]['overall']:.1f}%")