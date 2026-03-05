import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# -------------------------
# Patching
# -------------------------
def images_to_patch_vectors(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x: (B, 3, 32, 32)
    returns: (B, num_patches * patch_dim)  flattened patch features
    """
    B, C, H, W = x.shape
    p = patch_size
    patches = x.unfold(2, p, p).unfold(3, p, p)       # (B, C, H/p, W/p, p, p)
    patches = patches.permute(0, 2, 3, 1, 4, 5)       # (B, H/p, W/p, C, p, p)
    patches = patches.reshape(B, -1, C * p * p)        # (B, num_patches, patch_dim)
    feat = patches.reshape(B, -1)                      # (B, total_dim)
    return feat


# -------------------------
# Dataset wrapper
# -------------------------
class CIFAR10Patches(torch.utils.data.Dataset):
    def __init__(self, base_ds, patch_size: int):
        self.base_ds = base_ds
        self.patch_size = patch_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        feat = images_to_patch_vectors(x.unsqueeze(0), self.patch_size).squeeze(0)
        return feat, y


# -------------------------
# MLP — deeper and wider
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(2048, 1024, 512),
                 dropout=0.3, num_classes=10):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Train / eval
# -------------------------
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train(model, train_loader, val_loader, device,
          epochs=50, lr=1e-3, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()

        tr_acc = accuracy(model, train_loader, device)
        va_acc = accuracy(model, val_loader, device)
        avg_loss = running_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"epoch {ep:03d} | loss: {avg_loss:.4f} | "
              f"train: {tr_acc:.2f}% | val: {va_acc:.2f}% | "
              f"lr: {current_lr:.6f}")

        # Save best model based on validation accuracy
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  *** New best val acc: {best_val_acc:.2f}% ***")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"\nRestored best model (val acc: {best_val_acc:.2f}%)")

    return best_val_acc


@torch.no_grad()
def test_mlp(model, test_loader, device):
    model.eval()
    preds = []
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        preds.append(pred.cpu().numpy())
        correct += (pred == y).sum().item()
        total += y.numel()

    pred_labels = np.concatenate(preds, axis=0)
    acc_percent = 100.0 * correct / total
    return pred_labels, acc_percent


def main():
    # ── Config ──
    patch_size = 4
    batch_size = 256
    epochs = 50
    lr = 1e-3
    weight_decay = 5e-4
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── CIFAR-10 mean/std (well-known values) ──
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    # ── Transforms WITH augmentation for training ──
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # No augmentation for val/test
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # ── Load datasets ──
    train_full = datasets.CIFAR10(root="./data", train=True,
                                   download=True, transform=train_transform)
    # Need a separate copy with eval transform for validation
    train_full_eval = datasets.CIFAR10(root="./data", train=True,
                                        download=True, transform=eval_transform)
    test_base = datasets.CIFAR10(root="./data", train=False,
                                  download=True, transform=eval_transform)

    # ── 90/10 split ──
    n = len(train_full)
    n_val = int(0.1 * n)
    n_train = n - n_val

    gen = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(range(n), [n_train, n_val], generator=gen)

    train_subset = torch.utils.data.Subset(train_full, train_indices.indices)
    val_subset = torch.utils.data.Subset(train_full_eval, val_indices.indices)

    # ── Wrap with patching ──
    train_ds = CIFAR10Patches(train_subset, patch_size)
    val_ds = CIFAR10Patches(val_subset, patch_size)
    test_ds = CIFAR10Patches(test_base, patch_size)

    # num_workers=0 is safest on Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ──
    num_patches = (32 // patch_size) ** 2
    patch_dim = 3 * patch_size * patch_size
    input_dim = num_patches * patch_dim

    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Num patches: {num_patches}")
    print(f"Patch dim: {patch_dim}")
    print(f"Input dim: {input_dim}")
    print(f"Epochs: {epochs}")
    print()

    model = MLP(input_dim=input_dim,
                hidden_dims=(2048, 1024, 512),
                dropout=0.3).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()

    # ── Train ──
    best_val = train(model, train_loader, val_loader, device,
                     epochs=epochs, lr=lr, weight_decay=weight_decay)

    # ── Test ──
    pred_labels, test_acc = test_mlp(model, test_loader, device)
    print(f"\nFINAL TEST ACCURACY: {test_acc:.2f}%")

    # ── Save model (REQUIRED by spec) ──
    model_path = "q5_best_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # ── Save predictions (REQUIRED by spec) ──
    pred_path = "q5_test_pred_labels.npy"
    np.save(pred_path, pred_labels)
    print(f"Predictions saved to: {pred_path}")

    # ── Print design choices for report ──
    print()
    print("=" * 60)
    print("DESIGN CHOICES (for report)")
    print("=" * 60)
    print(f"  Patch size: {patch_size}x{patch_size} (captures local texture)")
    print(f"  Architecture: MLP with layers {input_dim}->2048->1024->512->10")
    print(f"  BatchNorm after each hidden layer (stabilises training)")
    print(f"  Dropout: 0.3 (regularisation to reduce overfitting)")
    print(f"  Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
    print(f"  Scheduler: CosineAnnealingLR over {epochs} epochs")
    print(f"  Augmentation: RandomHorizontalFlip, RandomCrop(32,pad=4), ColorJitter")
    print(f"  Normalisation: CIFAR-10 channel-wise mean/std")
    print(f"  Best validation accuracy: {best_val:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()