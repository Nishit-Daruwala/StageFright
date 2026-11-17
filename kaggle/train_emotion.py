"""Fine-tunes wav2vec2-base for speech emotion recognition on Kaggle (GPU).

Data: RAVDESS, CREMA-D, TESS and SAVEE, attached as Kaggle datasets (see kernel-metadata.json).
The split is by speaker: no test speaker is ever seen in training, so the reported
accuracy reflects new voices. Output, in /kaggle/working/emotion:
  - the model and feature extractor (copy this folder to models/emotion in the project)
  - metrics.json, confusion_matrix.png

Run from the project root:
  kaggle kernels push -p kaggle
  kaggle kernels status <username>/stagefright-emotion
  kaggle kernels output <username>/stagefright-emotion -p kaggle/output
"""

import json
import random
import re
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, get_linear_schedule_with_warmup

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
SR = 16000
MAX_S = 4.0
BASE = "facebook/wav2vec2-base"
LABELS = ["neutral", "happy", "sad", "angry", "fearful"]
EPOCHS, BATCH, LR = 6, 16, 3e-5
OUT = Path("/kaggle/working/emotion")
INPUT = Path("/kaggle/input")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------- file lists: (path, label, speaker)

def ravdess() -> list[tuple]:
    codes = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful"}
    out = {}
    for p in INPUT.rglob("03-01-*.wav"):                       # speech only; the dataset holds each file twice
        parts = p.stem.split("-")
        if len(parts) == 7 and parts[2] in codes:
            out[p.name] = (p, codes[parts[2]], f"rav{parts[6]}")
    return list(out.values())


def cremad() -> list[tuple]:
    codes = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry", "FEA": "fearful"}
    out = []
    for p in INPUT.rglob("*.wav"):
        m = re.fullmatch(r"(\d{4})_[A-Z]{3}_([A-Z]{3})_[A-Z]{2}", p.stem)
        if m and m.group(2) in codes:
            out.append((p, codes[m.group(2)], f"crema{m.group(1)}"))
    return out


def tess() -> list[tuple]:
    codes = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry", "fear": "fearful"}
    out = {}
    for p in INPUT.rglob("*.wav"):
        m = re.fullmatch(r"(OAF|YAF)_\w+_(\w+)", p.stem)
        if m and m.group(2).lower() in codes:
            out[p.name] = (p, codes[m.group(2).lower()], f"tess{m.group(1)}")
    return list(out.values())


def savee() -> list[tuple]:
    codes = {"n": "neutral", "h": "happy", "sa": "sad", "a": "angry", "f": "fearful"}
    out = []
    for p in INPUT.rglob("*.wav"):
        m = re.fullmatch(r"(DC|JE|JK|KL)_([a-z]+)\d+", p.stem)
        if m and m.group(2) in codes:
            out.append((p, codes[m.group(2)], f"savee{m.group(1)}"))
    return out


def speaker_split(items: list[tuple], test_frac: float = 0.2) -> tuple[list, list]:
    speakers = sorted({s for _, _, s in items})
    random.Random(SEED).shuffle(speakers)
    held_out = set(speakers[:max(1, int(len(speakers) * test_frac))])
    return [i for i in items if i[2] not in held_out], [i for i in items if i[2] in held_out]


# ---------------------------------------------------------------- data

class Clips(Dataset):
    def __init__(self, items: list[tuple], train: bool):
        self.items, self.train = items, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label, _ = self.items[i]
        x, _ = librosa.load(path, sr=SR, mono=True)
        x, _ = librosa.effects.trim(x, top_db=30)
        n = int(MAX_S * SR)
        if len(x) > n:
            start = random.randint(0, len(x) - n) if self.train else (len(x) - n) // 2
            x = x[start:start + n]
        if self.train:
            # Augmentation so the model copes with room microphones, not just studio recordings.
            x = x * random.uniform(0.5, 1.0)
            x = x + random.uniform(0.0, 0.01) * np.random.randn(len(x)).astype(np.float32)
        return x.astype(np.float32), LABELS.index(label)


def collate(extractor):
    def fn(batch):
        audio, labels = zip(*batch)
        enc = extractor(list(audio), sampling_rate=SR, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return fn


# ---------------------------------------------------------------- train

def evaluate(model, loader, device) -> tuple[list, list]:
    model.eval()
    truth, pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            truth += labels.tolist()
            pred += logits.argmax(-1).cpu().tolist()
    return truth, pred


def main() -> None:
    items = ravdess() + cremad() + tess() + savee()
    train_items, test_items = [], []
    for prefix in ("rav", "crema", "tess", "savee"):             # hold out speakers within each dataset
        tr, te = speaker_split([i for i in items if i[2].startswith(prefix)])
        train_items += tr
        test_items += te
    print(f"clips: {len(items)}  train: {len(train_items)}  test: {len(test_items)}")
    assert train_items and test_items, "no audio found - check the dataset sources"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = AutoFeatureExtractor.from_pretrained(BASE)
    model = AutoModelForAudioClassification.from_pretrained(
        BASE, num_labels=len(LABELS), label2id={l: i for i, l in enumerate(LABELS)},
        id2label={i: l for i, l in enumerate(LABELS)}).to(device)
    model.freeze_feature_encoder()                               # keep the low-level acoustic layers fixed

    train_dl = DataLoader(Clips(train_items, True), BATCH, shuffle=True, collate_fn=collate(extractor), num_workers=2)
    test_dl = DataLoader(Clips(test_items, False), BATCH, collate_fn=collate(extractor), num_workers=2)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(train_dl)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    OUT.mkdir(parents=True, exist_ok=True)
    best, history = 0.0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device, enabled=device == "cuda"):
                loss = model(**batch).loss
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            total += loss.item()
        truth, pred = evaluate(model, test_dl, device)
        acc, f1 = accuracy_score(truth, pred), f1_score(truth, pred, average="macro")
        history.append({"epoch": epoch, "loss": total / len(train_dl), "accuracy": acc, "macro_f1": f1})
        print(history[-1])
        if f1 > best:
            best = f1
            model.save_pretrained(OUT)
            extractor.save_pretrained(OUT)
            cm = confusion_matrix(truth, pred, labels=list(range(len(LABELS))))
            json.dump({"labels": LABELS, "accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist(),
                       "train_clips": len(train_items), "test_clips": len(test_items), "history": history},
                      open(OUT / "metrics.json", "w"), indent=2)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
            ax.set_yticks(range(len(LABELS)), LABELS)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Held-out speakers: accuracy {acc:.2f}, macro F1 {f1:.2f}")
            for r in range(len(LABELS)):
                for c in range(len(LABELS)):
                    ax.text(c, r, cm[r, c], ha="center", va="center",
                            color="white" if cm[r, c] > cm.max() / 2 else "black")
            fig.tight_layout()
            fig.savefig(OUT / "confusion_matrix.png", dpi=150)
            plt.close(fig)
    print(f"best macro F1: {best:.3f}")


if __name__ == "__main__":
    main()
