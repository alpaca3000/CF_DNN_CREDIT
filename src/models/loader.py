from .embed_mlp import EmbedMLP
from .classic_mlp import ClassicMLP
import torch
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