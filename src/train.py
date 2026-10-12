import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from src.data.pbc import get_dataloaders
from src.model import build_model


def run_epoch(model, dl, criterion, device, optimizer=None):
    """Una pasada por los datos. Con optimizer entrena; sin él, solo evalúa."""
    training = optimizer is not None
    model.train(training)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(dl, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_ft", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out", default="models/pbc_best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)

    train_dl, val_dl, test_dl, classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    history = []
    best_val = 0.0

    def fit(epochs, optimizer, phase):
        nonlocal best_val
        for ep in range(1, epochs + 1):
            tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
            vl, va = run_epoch(model, val_dl, criterion, device)
            print(f"[{phase}] época {ep}/{epochs} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
            history.append({"phase": phase, "epoch": ep, "train_loss": tl, "train_acc": ta,
                            "val_loss": vl, "val_acc": va})
            if va > best_val:
                best_val = va
                torch.save(model.state_dict(), args.out)

    # Fase 1: solo la capa final
    fit(args.epochs_head, torch.optim.Adam(model.classifier.parameters(), lr=1e-3), "cabeza")

    # Fase 2: fine-tuning, descongelamos los últimos bloques con learning rate bajo
    for p in model.features[-3:].parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    fit(args.epochs_ft, torch.optim.Adam(params, lr=1e-4), "fine-tuning")

    # Evaluación final en test con el mejor modelo
    model.load_state_dict(torch.load(args.out, map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_dl:
            y_pred += model(x.to(device)).argmax(1).cpu().tolist()
            y_true += y.tolist()

    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=classes).plot(
        xticks_rotation=45, cmap="Blues")
    plt.tight_layout()
    plt.savefig("reports/figures/pbc_confusion_matrix.png", dpi=150)

    with open("reports/pbc_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()