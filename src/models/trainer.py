# chứa hàm train_model() nhận model, train loader, val loader, test loader, config (số epoch, lr,...) và trả về model đã train xong cùng các metric trên val/test, tính toán loss, cập nhật trọng số, early stopping nếu không giảm loss tránh overfitting, lưu model tốt nhất dựa trên metric val/test (ví dụ AUC) để sau này load lại dùng cho giải thích mô hình. Hàm này sẽ được gọi trong main.py để huấn luyện các mô hình DNN. Các mô hình tree sẽ được huấn luyện bằng cách gọi trực tiếp hàm fit() của xgboost hoặc randomforest trong main.py.

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    return x


def _forward_with_target(
    model: nn.Module,
    batch: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hỗ trợ 2 kiểu batch:
    - ClassicMLP: (x, y)
    - EmbedMLP: (x_num, x_cat, y)
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError("Batch phải là tuple/list, ví dụ (x, y) hoặc (x_num, x_cat, y).")

    if len(batch) == 2:
        x, y = batch
        x = _to_device(x, device)
        y = _to_device(y, device).float().view(-1, 1)
        out = model(x)
        return out, y

    if len(batch) == 3:
        x_num, x_cat, y = batch
        x_num = _to_device(x_num, device)
        x_cat = _to_device(x_cat, device).long()
        y = _to_device(y, device).float().view(-1, 1)
        out = model(x_num, x_cat)
        return out, y

    raise ValueError("Batch không hợp lệ. Cần (x, y) hoặc (x_num, x_cat, y).")


def _run_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            probs, y = _forward_with_target(model, batch, device)
            loss = criterion(probs, y)

            if is_train:
                loss.backward()
                optimizer.step()

        preds = (probs >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    epoch_loss = total_loss / max(total_samples, 1)
    epoch_acc = total_correct / max(total_samples, 1)
    return epoch_loss, epoch_acc


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,  # BCE hoặc BCEWithLogits (nếu model trả sigmoid thì dùng BCE)
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Huấn luyện model với early stopping theo val_loss.

    Returns:
        model: model đã load best weights theo val_loss
        history: dict gồm train/val loss & acc mỗi epoch
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                break

    # load lại trọng số tốt nhất
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(
    model: nn.Module,
    test_loader: Iterable,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Trả về:
      - probs: xác suất dự báo (N,)
      - preds: nhãn dự báo nhị phân (N,)
    Hỗ trợ batch:
      - (x, y) hoặc (x_num, x_cat, y) hoặc (x,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_probs: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []

    for batch in test_loader:
        if isinstance(batch, (tuple, list)):
            if len(batch) == 1:
                x = _to_device(batch[0], device)
                probs = model(x)
            elif len(batch) == 2:
                # thường là (x, y): bỏ qua y khi predict
                x = _to_device(batch[0], device)
                probs = model(x)
            elif len(batch) == 3:
                x_num = _to_device(batch[0], device)
                x_cat = _to_device(batch[1], device).long()
                probs = model(x_num, x_cat)
            else:
                raise ValueError("Batch test không hợp lệ.")
        else:
            x = _to_device(batch, device)
            probs = model(x)

        probs = probs.view(-1)
        preds = (probs >= 0.5).long()

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_preds, dim=0)