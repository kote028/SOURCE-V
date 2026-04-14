"""
Training Script — Train all DeepShield modules on a labeled dataset.

Expected dataset directory layout:
    data/
      train/
        real/   *.mp4
        fake/   *.mp4
      val/
        real/   *.mp4
        fake/   *.mp4

Supports:
  - Standard training
  - Adversarial training (hardened against FGSM/PGD)
  - Saving checkpoints per module
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
#import tensorflow
from utils.video_pipeline import VideoPipeline
from modules.gaze_analyzer import GazeAnalyzer, GazeCNN, GazeLSTM
from modules.voice_analyzer import VoiceAnalyzer, SpectrogramCNN, AudioLSTM
from modules.emotion_behavioral_analyzer import EmotionBehavioralAnalyzer, EmotionCNN, BehavioralLSTM
from modules.adversarial_simulator import AdversarialSimulator


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Loads pre-extracted feature sequences from disk.
    Features are extracted once by run_feature_extraction() and cached as .npy.
    """

    def __init__(self, feature_dir: str, split: str = 'train'):
        self.samples: List[Tuple[str, int]] = []
        base = Path(feature_dir) / split
        for label, label_id in [('real', 0), ('fake', 1)]:
            path = base / label
            if path.exists():
                for f in path.glob('*.npy'):
                    self.samples.append((str(f), label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feat = np.load(path)
        return torch.from_numpy(feat).float(), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────
# Feature extraction (run once to cache)
# ─────────────────────────────────────────────────────────────────

def run_feature_extraction(video_root: str,
                            output_root: str,
                            seq_len: int = 32,
                            target_fps: float = 15.0):
    """
    Extract gaze CNN features from all videos in video_root and save as .npy.

    video_root layout: {split}/{real|fake}/*.mp4
    Saves to: output_root/{split}/{real|fake}/{name}.npy
    """
    pipeline = VideoPipeline(target_fps=target_fps, max_frames=seq_len * 5)
    gaze     = GazeAnalyzer(device='cpu', sequence_len=seq_len)

    LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                 157, 158, 159, 160, 161, 246]

    for split in ['train', 'val']:
        for label in ['real', 'fake']:
            src_dir = Path(video_root) / split / label
            out_dir = Path(output_root) / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.exists():
                continue

            videos = list(src_dir.glob('*.mp4')) + list(src_dir.glob('*.avi'))
            print(f"Extracting {split}/{label}: {len(videos)} videos")

            for vpath in tqdm(videos):
                out_path = out_dir / (vpath.stem + '.npy')
                if out_path.exists():
                    continue
                try:
                    pipe_out = pipeline.process(str(vpath))
                    frames   = pipe_out['frames'][:seq_len]
                    lm_seq   = pipe_out['landmarks_seq'][:seq_len]
                    if len(frames) < 4:
                        continue

                    # Extract CNN features for each frame
                    import cv2
                    feats = []
                    for frame, lm in zip(frames, lm_seq):
                        if lm is not None:
                            pts = lm[LEFT_EYE]
                            h, w = frame.shape[:2]
                            x1 = max(0, int(pts[:, 0].min() * w) - 6)
                            y1 = max(0, int(pts[:, 1].min() * h) - 6)
                            x2 = min(w, int(pts[:, 0].max() * w) + 6)
                            y2 = min(h, int(pts[:, 1].max() * h) + 6)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                crop = np.zeros((32, 64, 3), dtype=np.uint8)
                            else:
                                crop = cv2.resize(crop, (64, 32))
                        else:
                            crop = np.zeros((32, 64, 3), dtype=np.uint8)

                        t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
                        with torch.no_grad():
                            f = gaze.cnn(t.unsqueeze(0)).numpy()[0]
                        feats.append(f)

                    # Pad/truncate to seq_len
                    while len(feats) < seq_len:
                        feats.append(np.zeros(64, dtype=np.float32))
                    feats = feats[:seq_len]
                    np.save(out_path, np.stack(feats))

                except Exception as e:
                    print(f"  Skipped {vpath.name}: {e}")

    pipeline.cleanup()
    print("Feature extraction complete.")


# ─────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────

def train_gaze_lstm(feature_dir: str,
                    output_dir: str,
                    epochs: int = 50,
                    batch_size: int = 32,
                    lr: float = 1e-4,
                    adversarial: bool = False,
                    device: str = 'cpu') -> dict:
    """
    Train the GazeLSTM on pre-extracted features.

    Args:
        adversarial: If True, use adversarial training (Madry et al.)
                     to harden against FGSM/PGD attacks.
    """
    dev = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    train_ds = DeepfakeDataset(feature_dir, 'train')
    val_ds   = DeepfakeDataset(feature_dir, 'val')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = GazeLSTM(input_size=64, hidden=128, num_classes=2).to(dev)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    adv_sim = AdversarialSimulator(model, device=device) if adversarial else None

    history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x, y in tqdm(train_dl, desc=f'Epoch {epoch}/{epochs}', leave=False):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()

            if adversarial and adv_sim:
                loss = torch.tensor(adv_sim.adversarial_training_step(opt, x, y, epsilon=0.03))
                epoch_loss += loss.item() * len(x)
                continue

            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(x)

        sched.step()
        avg_loss = epoch_loss / max(1, len(train_ds))

        # Validation
        model.eval()
        correct = total = val_loss = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                val_loss += loss_fn(logits, y).item() * len(x)
                correct  += (logits.argmax(1) == y).sum().item()
                total    += len(y)

        val_acc  = correct / max(1, total)
        avg_vloss = val_loss / max(1, total)

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(avg_vloss)

        print(f"  Epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.3f} "
              f"| val_loss {avg_vloss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(output_dir, 'gaze_lstm_best.pt')
            torch.save(model.state_dict(), save_path)

    # Save final + history
    torch.save(model.state_dict(), os.path.join(output_dir, 'gaze_lstm_final.pt'))
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Weights saved to: {output_dir}")
    return history


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepShield Training')
    parser.add_argument('--mode',         choices=['extract', 'train'], required=True)
    parser.add_argument('--video_root',   default='data')
    parser.add_argument('--feature_dir',  default='data/features')
    parser.add_argument('--output_dir',   default='models/checkpoints')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--seq_len',      type=int,   default=32)
    parser.add_argument('--adversarial',  action='store_true',
                        help='Use adversarial training (Madry et al.)')
    parser.add_argument('--device',       default='cpu')
    args = parser.parse_args()

    if args.mode == 'extract':
        run_feature_extraction(args.video_root, args.feature_dir,
                                seq_len=args.seq_len)
    elif args.mode == 'train':
        train_gaze_lstm(
            feature_dir  = args.feature_dir,
            output_dir   = args.output_dir,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            lr           = args.lr,
            adversarial  = args.adversarial,
            device       = args.device,
        )
