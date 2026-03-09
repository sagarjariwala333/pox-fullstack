# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        MpoxNetV Safe Fine-Tuning — Google Colab                            ║
# ║        4 Classes: Chickenpox, Monkeypox, Measles, Normal                   ║
# ║        Fixes Chickenpox without breaking Monkeypox/Measles                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ──────────────────────────────────────────────────────────────────────────────
# CELL 1 — Check GPU
# ──────────────────────────────────────────────────────────────────────────────
# !nvidia-smi


# ──────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install dependencies
# ──────────────────────────────────────────────────────────────────────────────
# !pip install timm -q


# ──────────────────────────────────────────────────────────────────────────────
# CELL 3 — Mount Google Drive
# ──────────────────────────────────────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')


# ──────────────────────────────────────────────────────────────────────────────
# CELL 4 — Upload your .pt file (if not on Drive)
# ──────────────────────────────────────────────────────────────────────────────
# from google.colab import files
# files.upload()  # upload mpoxnet_v_fold2.pt


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 — CONFIG  ← Only edit this cell
# ══════════════════════════════════════════════════════════════════════════════
import os

# ── Your file paths ────────────────────────────────────────────────────────────
MODEL_PATH   = r"D:\pox\models\mpoxnet_v4class_fixed.pt"

# Original dataset — must have these exact subfolders:
#   Chickenpox/   ← 107 images
#   Monkeypox/    ← 279 images
#   Measles/      ← your count
#   Normal/       ← your count
ORIGINAL_DIR = r"D:\pox\dataset\MonkeypoxSkinImageDataset"

# MCVSLD Chickenpox folder (the 900 images from Mendeley)
# Point this to wherever you extracted the Chickenpox subfolder
MCVSLD_CHICKENPOX_DIR = r"D:\pox\dataset\MCVSLD\Chickenpox"

# Where to save the fixed model
SAVE_PATH = r"D:\pox\models\mpoxnet_v4class_fixed_fixed_3.pt"


# ── Class config ───────────────────────────────────────────────────────────────
# ⚠️  ORDER MUST match what your model was trained with
# If unsure — check your original training notebook's class_to_idx
CLASS_LABELS   = ["Chickenpox", "Measles", "Monkeypox", "Normal"]
NUM_CLASSES    = 4

CHICKENPOX_IDX = CLASS_LABELS.index("Chickenpox")   # 0
MONKEYPOX_IDX  = CLASS_LABELS.index("Monkeypox")    # 1
MEASLES_IDX    = CLASS_LABELS.index("Measles")       # 2
NORMAL_IDX     = CLASS_LABELS.index("Normal")        # 3

# ── Fine-tune config ───────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS       = 5         # reduced from 10 — model overshoots with more
VAL_SPLIT    = 0.2       # 80% train, 20% val
WEIGHT_DECAY = 1e-4

# Reduced from 1e-4 → 3e-5 to prevent boundary swinging too far
LR_HEAD = 3e-5

# You already merged 200 extra Chickenpox into your folder (107+200=307)
# No separate extra folder needed
EXTRA_CHICKENPOX_LIMIT = 0

# Safety floors — updated based on post-finetune results
# Monkeypox dropped to 49% last time → trigger earlier to catch the drop
MONKEYPOX_FLOOR = 75.0
MEASLES_FLOOR   = 72.0

# Early stopping patience — reduced, stop faster if no improvement
PATIENCE = 3


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU   : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU found — training will be very slow")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 — MODEL ARCHITECTURE (exact same as your training code)
# ══════════════════════════════════════════════════════════════════════════════
class CrossAttentionGate(nn.Module):
    def __init__(self, dim=1024, hidden=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
            nn.Softmax(dim=1),
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
    def forward(self, x, return_gate=False):
        tokens = self.deit.forward_features(x)
        G = self.proj_g(torch.cat([tokens[:, 0], tokens[:, 1]], dim=1))
        L = self.proj_l(self.effnet(x))
        F, alpha, beta = self.gate(G, L)
        logits = self.head(F)
        return (logits, alpha, beta) if return_gate else logits


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 — LOAD FOLD2 WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# ⚠️  Force num_classes=4 regardless of what's in checkpoint
# Your checkpoint may say 6 but your actual classes are 4
model = MpoxNetV(num_classes=NUM_CLASSES).to(DEVICE)

state = checkpoint['model_state_dict']

# Handle head size mismatch if checkpoint was 6-class
# Replace the final linear layer with 4-class output
try:
    model.load_state_dict(state, strict=True)
    print(f"✅ Loaded weights (strict) — checkpoint matches 4 classes")
except RuntimeError:
    print("⚠️  Head size mismatch detected (checkpoint=6 classes, target=4 classes)")
    print("   Loading backbone weights only, reinitializing head...")
    # Load everything except the final classification layer
    filtered = {
        k: v for k, v in state.items()
        if not k.startswith("head.3")   # skip final Linear(512, num_classes)
    }
    model.load_state_dict(filtered, strict=False)
    print("✅ Backbone loaded, head reinitialized for 4 classes")

print(f"   num_classes = {NUM_CLASSES}")
print(f"   class order = {CLASS_LABELS}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 — FREEZE BACKBONE (Key protection against forgetting)
# ══════════════════════════════════════════════════════════════════════════════

# Freeze DeiT and EfficientNet completely
# They already learned good visual features — don't touch them
for param in model.deit.parameters():
    param.requires_grad = False
for param in model.effnet.parameters():
    param.requires_grad = False

# Only these layers will be updated during fine-tuning
trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
total_params = sum(p.numel() for p in model.parameters())

print(f"✅ Backbone FROZEN")
print(f"   Trainable params : {trainable_params:,}  ({100*trainable_params/total_params:.1f}% of total)")
print(f"   Frozen params    : {total_params - trainable_params:,}")
print(f"   Only training    : proj_g, proj_l, gate, head")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 — TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

# Heavy augmentation ONLY for Chickenpox
chickenpox_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Light augmentation for all other classes
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# No augmentation for validation — clean evaluation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 — DATASET
# ══════════════════════════════════════════════════════════════════════════════

class SkinDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples  = samples
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.is_train and label == CHICKENPOX_IDX:
            image = chickenpox_transform(image)
        elif self.is_train:
            image = base_transform(image)
        else:
            image = val_transform(image)
        return image, label


def load_original_samples(data_dir):
    """Load images from original dataset folders."""
    samples = []
    for label_idx, class_name in enumerate(CLASS_LABELS):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"⚠️  Folder not found — skipping: {class_dir}")
            continue
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]
        for fname in files:
            samples.append((os.path.join(class_dir, fname), label_idx))
        print(f"   {class_name:<15}: {len(files)} images")
    return samples


def load_extra_chickenpox(mcvsld_dir, limit):
    """
    Load extra Chickenpox images from MCVSLD.
    Only takes `limit` images — enough to balance, not flood.
    """
    if not os.path.exists(mcvsld_dir):
        print(f"⚠️  MCVSLD folder not found: {mcvsld_dir}")
        print("    Fine-tuning will continue with original data only")
        print("    Chickenpox will still be oversampled via WeightedSampler")
        return []

    all_files = [
        f for f in os.listdir(mcvsld_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    random.seed(42)
    selected = random.sample(all_files, min(limit, len(all_files)))
    samples  = [(os.path.join(mcvsld_dir, f), CHICKENPOX_IDX) for f in selected]
    print(f"   {'Chickenpox(MCVSLD)':<15}: {len(samples)} extra images added")
    return samples


# Load
print("\n📂 Loading original dataset:")
original_samples = load_original_samples(ORIGINAL_DIR)

print("\n📂 Loading extra Chickenpox from MCVSLD:")
extra_chickenpox = load_extra_chickenpox(MCVSLD_CHICKENPOX_DIR, EXTRA_CHICKENPOX_LIMIT)

# Split original into train/val FIRST
labels_only = [s[1] for s in original_samples]
train_original, val_samples = train_test_split(
    original_samples,
    test_size=VAL_SPLIT,
    stratify=labels_only,
    random_state=42
)

# Add extra chickenpox to TRAIN ONLY — val stays clean and unmodified
train_samples = train_original + extra_chickenpox
random.seed(42)
random.shuffle(train_samples)

# Print final distribution
print("\n📊 Final distribution:")
train_counts = Counter([s[1] for s in train_samples])
val_counts   = Counter([s[1] for s in val_samples])
print(f"  {'Class':<20} {'Train':>8} {'Val':>8}")
print("  " + "─" * 38)
for idx, name in enumerate(CLASS_LABELS):
    marker = " ← was 107, now balanced" if idx == CHICKENPOX_IDX else ""
    print(f"  {name:<20} {train_counts.get(idx,0):>8} {val_counts.get(idx,0):>8}{marker}")
print(f"  {'TOTAL':<20} {len(train_samples):>8} {len(val_samples):>8}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 — WEIGHTED SAMPLER + LOSS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

sample_labels  = [s[1] for s in train_samples]
class_counts   = Counter(sample_labels)
total_samples  = len(sample_labels)

# WeightedRandomSampler — ensures every batch sees balanced classes
sample_weights = [total_samples / class_counts[l] for l in sample_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

print("\n⚖️  Sampler weights:")
for idx, name in enumerate(CLASS_LABELS):
    if idx in class_counts:
        print(f"   {name:<20}: {total_samples / class_counts[idx]:.2f}x per batch")

train_loader = DataLoader(
    SkinDataset(train_samples, is_train=True),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=False
)
val_loader = DataLoader(
    SkinDataset(val_samples, is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

# ── Manual loss weights — balanced, with slight Monkeypox boost to recover ────
# Last run: Chickenpox weight was too high → Monkeypox collapsed to 49%
# Now: equal weights with slight Monkeypox boost to recover
loss_weights = torch.tensor([
    1.0,   # Chickenpox — equal (has 307 images now, no longer starved)
    1.0,   # Measles    — equal
    1.3,   # Monkeypox  — slight boost to recover from 49% collapse
    1.0,   # Normal     — equal
])

print("\n🎯 Loss weights (manual — balanced to prevent overcorrection):")
for idx, name in enumerate(CLASS_LABELS):
    print(f"   {name:<20}: {loss_weights[idx]:.1f}")

criterion = nn.CrossEntropyLoss(
    weight=loss_weights.to(DEVICE),
    label_smoothing=0.05   # reduced from 0.1 — less smoothing = sharper boundaries
)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 13 — OPTIMIZER (head only — backbone is frozen)
# ══════════════════════════════════════════════════════════════════════════════

optimizer = torch.optim.AdamW([
    {'params': model.proj_g.parameters()},
    {'params': model.proj_l.parameters()},
    {'params': model.gate.parameters()},
    {'params': model.head.parameters()},
], lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-7
)

print(f"✅ Optimizer ready — LR={LR_HEAD} (head only, backbone frozen)")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 14 — FINE-TUNE LOOP
# ══════════════════════════════════════════════════════════════════════════════

best_val_acc    = 0.0
best_chickenpox = 0.0
patience_ctr    = 0
stop_reason     = None

print(f"\n🚀 Fine-tuning {EPOCHS} epochs | Backbone FROZEN | Head only")
print(f"   Safety floors: Monkeypox≥{MONKEYPOX_FLOOR}% | Measles≥{MEASLES_FLOOR}%")
print(f"   Watching: Chkpox→Mpox confusion column (target: 0)\n")

header = f"{'Ep':<5} {'Loss':<8} {'Train':<8} {'Val':<8} {'Mpox%':<9} {'Chkpox%':<10} {'Measles%':<11} {'Normal%':<10} {'Confusion'}"
print(header)
print("─" * len(header))

for epoch in range(1, EPOCHS + 1):

    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    train_loss = train_correct = train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds          = outputs.argmax(dim=1)
        train_loss    += loss.item()
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    scheduler.step()

    # ── Validate ───────────────────────────────────────────────────────────────
    model.eval()
    class_correct  = [0] * NUM_CLASSES
    class_total    = [0] * NUM_CLASSES
    chkpox_as_mpox = 0   # the specific confusion we are fixing

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)

            for pred, label in zip(preds, labels):
                l = label.item()
                p = pred.item()
                class_total[l] += 1
                if p == l:
                    class_correct[l] += 1
                if l == CHICKENPOX_IDX and p == MONKEYPOX_IDX:
                    chkpox_as_mpox += 1

    train_acc   = train_correct / train_total * 100
    val_acc     = sum(class_correct) / max(sum(class_total), 1) * 100
    mpox_acc    = class_correct[MONKEYPOX_IDX]  / max(class_total[MONKEYPOX_IDX],  1) * 100
    chkpox_acc  = class_correct[CHICKENPOX_IDX] / max(class_total[CHICKENPOX_IDX], 1) * 100
    measles_acc = class_correct[MEASLES_IDX]    / max(class_total[MEASLES_IDX],    1) * 100
    normal_acc  = class_correct[NORMAL_IDX]     / max(class_total[NORMAL_IDX],     1) * 100

    print(
        f"{epoch:02d}/{EPOCHS:<3} "
        f"{train_loss/len(train_loader):<8.3f} "
        f"{train_acc:<8.1f} "
        f"{val_acc:<8.1f} "
        f"{mpox_acc:<9.1f} "
        f"{chkpox_acc:<10.1f} "
        f"{measles_acc:<11.1f} "
        f"{normal_acc:<10.1f} "
        f"{chkpox_as_mpox}"
    )

    # ── Safety Floor Check ─────────────────────────────────────────────────────
    if mpox_acc < MONKEYPOX_FLOOR:
        stop_reason = f"⛔ Monkeypox dropped below floor ({mpox_acc:.1f}% < {MONKEYPOX_FLOOR}%)"
        print(stop_reason)
        print("   Rolling back to last saved checkpoint...")
        model.load_state_dict(
            torch.load(SAVE_PATH, map_location=DEVICE)['model_state_dict']
        )
        break

    if measles_acc < MEASLES_FLOOR:
        stop_reason = f"⛔ Measles dropped below floor ({measles_acc:.1f}% < {MEASLES_FLOOR}%)"
        print(stop_reason)
        print("   Rolling back to last saved checkpoint...")
        model.load_state_dict(
            torch.load(SAVE_PATH, map_location=DEVICE)['model_state_dict']
        )
        break

    # ── Save Best ──────────────────────────────────────────────────────────────
    # Prioritize Chickenpox improvement, but only if other classes are healthy
    if chkpox_acc > best_chickenpox or val_acc > best_val_acc:
        best_val_acc    = val_acc
        best_chickenpox = chkpox_acc
        patience_ctr    = 0
        torch.save({
            'model_state_dict' : model.state_dict(),
            'num_classes'      : NUM_CLASSES,
            'class_labels'     : CLASS_LABELS,
            'epoch'            : epoch,
            'val_acc'          : val_acc,
            'chickenpox_acc'   : chkpox_acc,
            'monkeypox_acc'    : mpox_acc,
            'measles_acc'      : measles_acc,
            'normal_acc'       : normal_acc,
        }, SAVE_PATH)
        print(f"   💾 Saved — val={val_acc:.1f}% | chkpox={chkpox_acc:.1f}% | mpox={mpox_acc:.1f}%")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n⏹️  Early stopping — no improvement for {PATIENCE} epochs")
            break

# ── Final Summary ──────────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  ✅ Fine-tuning complete")
print(f"  Best val accuracy  : {best_val_acc:.1f}%")
print(f"  Best chickenpox    : {best_chickenpox:.1f}%  (was 31.5%)")
print(f"  Saved to           : {SAVE_PATH}")
if stop_reason:
    print(f"  Stopped because    : {stop_reason}")
print(f"{'═'*60}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 15 — INFERENCE CODE (copy this to your API)
# ══════════════════════════════════════════════════════════════════════════════
import io
from torchvision import transforms
from PIL import Image

# Reload fixed model for inference
infer_checkpoint = torch.load(SAVE_PATH, map_location=torch.device('cpu'), weights_only=False)
infer_model      = MpoxNetV(num_classes=NUM_CLASSES)
infer_model.load_state_dict(infer_checkpoint['model_state_dict'])
infer_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CONFIDENCE_THRESHOLD = 0.60   # below this → uncertain

def predict(image_bytes: bytes) -> dict:
    image        = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = infer_model(input_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).tolist()

    prob_dict   = {cls: round(p, 4) for cls, p in zip(CLASS_LABELS, probs)}
    top_idx     = probs.index(max(probs))
    raw_label   = CLASS_LABELS[top_idx]
    confidence  = round(probs[top_idx], 4)
    uncertain   = confidence < CONFIDENCE_THRESHOLD

    # Extra guard: if Monkeypox and Chickenpox are very close → flag uncertain
    mpox_p  = prob_dict["Monkeypox"]
    chkp_p  = prob_dict["Chickenpox"]
    if abs(mpox_p - chkp_p) < 0.15 and raw_label in ("Monkeypox", "Chickenpox"):
        uncertain  = True
        raw_label  = "Uncertain - Monkeypox or Chickenpox"

    return {
        "label"        : raw_label,
        "confidence"   : confidence,
        "probabilities": prob_dict,
        "uncertain"    : uncertain,
    }

print("✅ Inference function ready")
print(f"   Classes : {CLASS_LABELS}")
print(f"   Threshold: {CONFIDENCE_THRESHOLD}")