# file chứa các hàm khởi tạo và huấn luyện nhanh mô hình xgboost và randomforest

from __future__ import annotations

from typing import Optional, Dict, Any

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_xgboost_model(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    random_state: int = 42,
    n_jobs: int = -1,
    scale_pos_weight: Optional[float] = None,
) -> XGBClassifier:
    """
    Khởi tạo XGBoost cho bài toán phân loại nhị phân.
    """
    params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    return XGBClassifier(**params)

def get_random_forest_model(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    class_weight: Optional[str] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Khởi tạo RandomForest cho bài toán phân loại nhị phân.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

def build_random_forest(cfg: Dict[str, Any]) -> Any:
    return get_random_forest_model(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=cfg.get("max_depth", None),
        min_samples_split=int(cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
        max_features=cfg.get("max_features", "sqrt"),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


def build_xgboost(cfg: Dict[str, Any]) -> Any:
    return get_xgboost_model(
        n_estimators=int(cfg.get("n_estimators", 300)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("max_depth", 4)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        min_child_weight=float(cfg.get("min_child_weight", 1.0)),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", 1)),
    )

# ====================
# Search Spaces
# ====================

def get_xgboost_search_space() -> Dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 0.5, "high": 2.0},
        "min_child_weight": {"type": "float", "low": 1.0, "high": 5.0},
        "random_state": 42,
        "n_jobs": -1,
    }

def get_random_forest_search_space() -> Dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": {"type": "int", "low": 2, "high": 10, "step": 2},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 5, "step": 1},
        "max_features": ["sqrt", "log2"],
        "random_state": 42,
        "n_jobs": -1,
    }
