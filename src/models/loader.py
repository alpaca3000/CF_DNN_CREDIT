from .embed_mlp import EmbedMLP
from .classic_mlp import ClassicMLP
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any
import json
import pickle
from pathlib import Path
import sys
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"


def _load_best_config(dataset: str, model_name: str) -> dict[str, Any]:
    cfg_path = RESULTS_DIR / dataset / "best_configs.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy best config tại: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    best_cfg = cfg_json.get(model_name, {}).get("best_config", {})
    if not best_cfg:
        raise ValueError(f"best_configs.json không có cấu hình cho {model_name}.")
    return best_cfg


def load_embed_model(dataset: str, device: torch.device) -> EmbedMLP:
    model_path = MODELS_DIR / dataset / "embed_mlp_best.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model embed_mlp tại: {model_path}")

    best_cfg = _load_best_config(dataset, "embed_mlp")

    model = EmbedMLP(
        input_num_dim=int(best_cfg["input_num_dim"]),
        cat_dims=list(best_cfg["cat_dims"]),
        emb_dims=best_cfg.get("emb_dims", None),
        hidden_dims=(int(best_cfg["hidden_h1"]), int(best_cfg["hidden_h2"])),
        dropout=float(best_cfg.get("dropout", 0.3)),
    )

    try:
        state_obj = torch.load(model_path, map_location=device, weights_only=True)  # pyright: ignore[reportUnknownMemberType]
    except TypeError:
        state_obj = torch.load(model_path, map_location=device)  # pyright: ignore[reportUnknownMemberType]

    if not isinstance(state_obj, dict):
        raise ValueError("embed_mlp_best.pkl không phải state_dict như mong đợi.")

    state = cast(dict[str, torch.Tensor], state_obj)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_classic_model(dataset: str, device: torch.device) -> ClassicMLP:
    model_path = MODELS_DIR / dataset / "classic_mlp_best.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model classic_mlp tại: {model_path}")

    best_cfg = _load_best_config(dataset, "classic_mlp")

    model = ClassicMLP(
        input_dim=int(best_cfg["input_dim"]),
        hidden_dims=(int(best_cfg["hidden_h1"]), int(best_cfg["hidden_h2"])),
        dropout=float(best_cfg.get("dropout", 0.3)),
    )

    try:
        state_obj = torch.load(model_path, map_location=device, weights_only=True)  # pyright: ignore[reportUnknownMemberType]
    except TypeError:
        state_obj = torch.load(model_path, map_location=device)  # pyright: ignore[reportUnknownMemberType]

    if not isinstance(state_obj, dict):
        raise ValueError("classic_mlp_best.pkl không phải state_dict như mong đợi.")

    state = cast(dict[str, torch.Tensor], state_obj)

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_xgboost_model(dataset: str) -> Any:
    model_path = MODELS_DIR / dataset / "xgboost_best.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model xgboost tại: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_random_forest_model(dataset: str) -> Any:
    model_path = MODELS_DIR / dataset / "random_forest_best.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model random_forest tại: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_model(model_name: str, dataset: str, device: torch.device | None = None) -> Any:
    if model_name == "embed_mlp":
        if device is None:
            raise ValueError("device là bắt buộc khi load embed_mlp.")
        return load_embed_model(dataset, device)
    if model_name == "classic_mlp":
        if device is None:
            raise ValueError("device là bắt buộc khi load classic_mlp.")
        return load_classic_model(dataset, device)
    if model_name == "xgboost":
        return load_xgboost_model(dataset)
    if model_name == "random_forest":
        return load_random_forest_model(dataset)
    raise ValueError(f"Model name '{model_name}' không được hỗ trợ.")

# def create_dataloaders(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     batch_size: int = 64,
# ) -> Dict[str, DataLoader]:
#     """Create PyTorch DataLoaders."""
#     X_train_t = torch.from_numpy(X_train).float()
#     y_train_t = torch.from_numpy(y_train).long()
    
#     X_val_t = torch.from_numpy(X_val).float()
#     y_val_t = torch.from_numpy(y_val).long()
    
#     X_test_t = torch.from_numpy(X_test).float()
#     y_test_t = torch.from_numpy(y_test).long()
    
#     train_ds = TensorDataset(X_train_t, y_train_t)
#     val_ds = TensorDataset(X_val_t, y_val_t)
#     test_ds = TensorDataset(X_test_t, y_test_t)
    
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
#     return {
#         "train_loader": train_loader,
#         "val_loader": val_loader,
#         "test_loader": test_loader,
#     }

def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def _split_num_cat(X_all: np.ndarray, cat_idxs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not cat_idxs:
        return X_all.astype(np.float32), np.zeros((X_all.shape[0], 0), dtype=np.int64)
    x_cat = X_all[:, cat_idxs].astype(np.int64)
    num_idxs = [i for i in range(X_all.shape[1]) if i not in cat_idxs]
    x_num = X_all[:, num_idxs].astype(np.float32)
    return x_num, x_cat

# def create_embedding_dataloaders(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: np.ndarray,
#     y_val: np.ndarray,
#     X_test: np.ndarray,
#     y_test: np.ndarray,
#     cat_idxs: list[int],
#     batch_size: int = 64,
# ) -> Dict[str, DataLoader]:
#     xnum_tr, xcat_tr = _split_num_cat(X_train, cat_idxs)
#     xnum_va, xcat_va = _split_num_cat(X_val, cat_idxs)
#     xnum_te, xcat_te = _split_num_cat(X_test, cat_idxs)

#     train_ds = TensorDataset(
#         torch.from_numpy(xnum_tr).float(),
#         torch.from_numpy(xcat_tr).long(),
#         torch.from_numpy(y_train).long(),
#     )
#     val_ds = TensorDataset(
#         torch.from_numpy(xnum_va).float(),
#         torch.from_numpy(xcat_va).long(),
#         torch.from_numpy(y_val).long(),
#     )
#     test_ds = TensorDataset(
#         torch.from_numpy(xnum_te).float(),
#         torch.from_numpy(xcat_te).long(),
#         torch.from_numpy(y_test).long(),
#     )

#     return {
#         "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
#         "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
#         "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
#     }

def create_embedding_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    cat_idxs: list[int],
    batch_size: int = 64,
    shuffle: bool = False,
) -> Dict[str, DataLoader]:
    xnum, xcat = _split_num_cat(X, cat_idxs)

    ds = TensorDataset(
        torch.from_numpy(xnum).float(),
        torch.from_numpy(xcat).long(),
        torch.from_numpy(y).long(),
    )

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)