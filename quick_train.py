"""
quick_train.py  —  Train MobileNetV2 deepfake classifier on FF++ dataset.

Data layout expected:
    FF++/
        real/   *.mp4   (real videos)
        fake/   *.mp4   (deepfake videos)

Output:
    models/mobilenet_deepfake.pt

Runs in ~15-20 min on CPU (uses 8 frames per video, 30 videos per class).
GPU will be used automatically if available.
"""

import os, cv2, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_DIR       = "FF++"          # folder with real/ and fake/ subfolders
MODEL_OUT      = "models/mobilenet_deepfake.pt"
FRAMES_PER_VID = 8               # face crops to extract per video
IMG_SIZE       = 112             # input resolution (112 is fast, 224 is better)
MAX_VIDEOS     = 80              # per class (reduce if running out of time)
EPOCHS         = 6               # increase for better accuracy
BATCH_SIZE     = 16
LR             = 3e-4
VAL_SPLIT      = 0.15            # 15 % of data held out for validation

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Face extractor ────────────────────────────────────────────────────
_cascade = None
def get_cascade():
    global _cascade
    if _cascade is None:
        p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(p)
    return _cascade

def crop_face(frame, size=IMG_SIZE):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = get_cascade().detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        pad = int(0.15 * max(w, h))
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(frame.shape[1], x+w+pad)
        y2 = min(frame.shape[0], y+h+pad)
        crop = frame[y1:y2, x1:x2]
    else:
        h, w = frame.shape[:2]
        sz = min(h, w)
        crop = frame[(h-sz)//2:(h-sz)//2+sz, (w-sz)//2:(w-sz)//2+sz]
    return cv2.resize(crop, (size, size))

def extract_frames(video_path, n=FRAMES_PER_VID):
    cap   = cv2.VideoCapture(video_path)
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    idxs  = np.linspace(0, total-1, n, dtype=int)
    crops = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            crops.append(crop_face(frame))
    cap.release()
    return crops

def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb).permute(2, 0, 1).float()

# ── Dataset ───────────────────────────────────────────────────────────
class FrameDataset(Dataset):
    def __init__(self, items):  # items = [(tensor, label), ...]
        self.items = items
    def __len__(self):  return len(self.items)
    def __getitem__(self, i): return self.items[i]

def build_dataset(data_dir, max_per_class=MAX_VIDEOS):
    print(f"\n[build_dataset] Scanning {data_dir}...")
    all_items = []
    for label_name, label_id in [("real", 0), ("fake", 1)]:
        folder = Path(data_dir) / label_name
        videos = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.avi")))
        random.shuffle(videos)
        videos = videos[:max_per_class]
        print(f"  {label_name}: {len(videos)} videos -> extracting {FRAMES_PER_VID} frames each...")
        for i, vp in enumerate(videos):
            try:
                crops = extract_frames(str(vp))
                for crop in crops:
                    all_items.append((to_tensor(crop), label_id))
                if (i+1) % 10 == 0:
                    print(f"    [{label_name}] {i+1}/{len(videos)} done")
            except Exception as e:
                print(f"    SKIP {vp.name}: {e}")
    random.shuffle(all_items)
    print(f"  Total samples: {len(all_items)} frames")
    return all_items

# ── Model ─────────────────────────────────────────────────────────────
def build_model(num_classes=2):
    m = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze early layers for fast training
    for i, layer in enumerate(m.features):
        if i < 10:  # freeze first 10 of 18 layers
            for p in layer.parameters():
                p.requires_grad = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    return m

# ── Train loop ────────────────────────────────────────────────────────
def train():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[quick_train] Device: {device}")

    # Build dataset
    t0 = time.time()
    items = build_dataset(DATA_DIR)
    split = int(len(items) * (1 - VAL_SPLIT))
    train_items, val_items = items[:split], items[split:]
    print(f"\n  Train: {len(train_items)} | Val: {len(val_items)}")
    print(f"  Frame extraction took {time.time()-t0:.1f}s")

    train_dl = DataLoader(FrameDataset(train_items), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(FrameDataset(val_items),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = build_model().to(device)
    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, steps_per_epoch=len(train_dl), epochs=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        model.train()
        ep_loss = correct = total = 0
        t_ep = time.time()

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item() * len(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += len(y)

        train_acc = correct / max(1, total)

        # Validation
        model.eval()
        vcorrect = vtotal = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                vcorrect += (preds == y).sum().item()
                vtotal   += len(y)

        val_acc = vcorrect / max(1, vtotal)
        elapsed = time.time() - t_ep
        print(f"  Epoch {epoch}/{EPOCHS} | loss {ep_loss/max(1,total):.4f} | "
              f"train_acc {train_acc:.3f} | val_acc {val_acc:.3f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"  -> Saved best model (val_acc={val_acc:.3f})")

    print(f"\n[DONE] Best val accuracy: {best_acc:.3f}")
    print(f"Model saved to: {MODEL_OUT}")
    print("\nRestart main.py to use the trained model!")

if __name__ == "__main__":
    train()
