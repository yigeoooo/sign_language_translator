# trainer.py
import os
import json
import math
import time
import glob
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_definition import ModelFactory
from data_preprocessor import HandGesturePreprocessor


@dataclass
class TrainConfig:
    model_type: str = "cnn_lstm"        # 可选: "lstm" / "cnn_lstm"
    epochs: int = 60
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    use_weighted_sampler: bool = True   # 是否使用均衡采样
    early_stop_patience: int = 12
    plateau_patience: int = 6
    plateau_factor: float = 0.5
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    processed_pkl: str = ""             # 留空则自动找最新的
    save_dir: str = "data/models"
    seed: int = 42


def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_pkl(processed_dir: str = "data/processed") -> str:
    os.makedirs(processed_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(processed_dir, "*.pkl")))
    return files[-1] if files else ""


def bincount_safe(y: np.ndarray, num_classes: int) -> np.ndarray:
    bc = np.bincount(y, minlength=num_classes)
    bc[bc == 0] = 1  # 防止除0
    return bc


def make_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = bincount_safe(y, num_classes)
    weights = 1.0 / counts.astype(np.float32)
    # 归一化到和为 num_classes（更稳）
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def describe_distribution(y: np.ndarray, label_decoder: dict):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print("\n[数据分布] 训练集：")
    for cls, cnt in zip(unique, counts):
        name = label_decoder.get('gesture', {}).get(int(cls), str(cls))
        print(f"  类别 {cls} ({name:>8}): {cnt:>4d} 样本 ({cnt/total*100:5.1f}%)")
    if counts.size:
        print(f"  不平衡比: {counts.max() / max(1, counts.min()):.2f}")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def build_loaders(X_train, y_train, X_val, y_val, cfg: TrainConfig):
    # 期望 X_* 形状: (N, T, F) 或 (N, F)；y_* 为 (N,)
    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    # 采样器（均衡采样）
    if cfg.use_weighted_sampler:
        class_counts = np.bincount(y_train, minlength=int(y_train.max()) + 1)
        class_counts[class_counts == 0] = 1
        sample_weights = np.array([1.0 / class_counts[y] for y in y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                                  num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(val_ds, batch_size=max(64, cfg.batch_size), shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter, acc_meter, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = yb.size(0)
        loss_meter += loss.item() * bs
        acc_meter += accuracy(logits.detach(), yb) * bs
        n += bs
    return loss_meter / max(1, n), acc_meter / max(1, n)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_meter, acc_meter, n = 0.0, 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        bs = yb.size(0)
        loss_meter += loss.item() * bs
        acc_meter += accuracy(logits, yb) * bs
        n += bs
    return loss_meter / max(1, n), acc_meter / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="cnn_lstm", choices=["lstm", "cnn_lstm"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_weighted_sampler", action="store_true", help="关闭均衡采样")
    parser.add_argument("--processed_pkl", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="data/models")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_weighted_sampler=(not args.no_weighted_sampler),
        processed_pkl=args.processed_pkl,
        save_dir=args.save_dir,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 1) 加载预处理后的数据
    processed_path = cfg.processed_pkl or find_latest_pkl("data/processed")
    if not processed_path:
        raise FileNotFoundError("未找到 processed 数据文件，请先运行数据预处理。")

    print(f"[数据] 使用 processed: {os.path.basename(processed_path)}")
    preproc = HandGesturePreprocessor()
    splits = preproc.load_processed_data(processed_path)

    # 你在预处理里通常会有这些键：
    # X_train, y_gesture_train, X_val, y_gesture_val
    X_train = splits["X_train"]
    y_train = splits["y_gesture_train"]
    X_val = splits["X_val"]
    y_val = splits["y_gesture_val"]

    num_classes = len(preproc.label_decoder["gesture"])
    input_dim = X_train.shape[-1]
    print(f"[数据] 训练集: {X_train.shape}, 验证集: {X_val.shape}, 类别数: {num_classes}, 特征维: {input_dim}")
    describe_distribution(y_train, preproc.label_decoder)

    # 2) 构建数据加载器
    train_loader, val_loader = build_loaders(X_train, y_train, X_val, y_val, cfg)

    # 3) 创建模型
    model = ModelFactory.create_model(
        cfg.model_type,
        input_dim=input_dim,
        num_classes=num_classes
    ).to(cfg.device)

    # 4) 构建损失（类别权重），去掉 label smoothing
    class_weights = make_class_weights(y_train, num_classes).to(cfg.device)
    print(f"[损失] 类别权重: {class_weights.detach().cpu().numpy().round(3).tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5) 优化器 & 调度器
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                  patience=cfg.plateau_patience,
                                  factor=cfg.plateau_factor,
                                  verbose=True)

    # 6) 训练循环
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, cfg.device)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d}/{cfg.epochs} | "
              f"train_loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
              f"val_loss: {va_loss:.4f} acc: {va_acc:.3f} | "
              f"{dt:.1f}s")

        # 调整学习率（看验证准确率）
        scheduler.step(va_acc)

        # 早停逻辑
        if va_acc > best_val_acc + 1e-4:
            best_val_acc = va_acc
            best_state = {
                "model_type": cfg.model_type,
                "model_state_dict": model.state_dict(),
                "metrics": {
                    "best_val_acc": best_val_acc,
                    "last_val_loss": float(va_loss),
                    "epoch": epoch
                },
                "train_config": asdict(cfg)
            }
            torch.save(best_state, os.path.join(cfg.save_dir, "best_model.pth"))
            print(f"  ↑ 保存最佳模型 (val_acc={best_val_acc:.3f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"早停：{cfg.early_stop_patience} 个 epoch 未提升。")
                break

    # 7) 结束与兜底保存
    if best_state is None:
        best_state = {
            "model_type": cfg.model_type,
            "model_state_dict": model.state_dict(),
            "metrics": {"best_val_acc": best_val_acc},
            "train_config": asdict(cfg)
        }
        torch.save(best_state, os.path.join(cfg.save_dir, "best_model.pth"))
        print("未出现更优验证准确率，已保存当前模型为 best_model.pth")

    print("\n训练完成。最佳验证准确率：", f"{best_val_acc:.3f}")
    print("模型保存路径：", os.path.join(cfg.save_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
