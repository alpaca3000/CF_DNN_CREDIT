# file chứa các hàm khởi tạo và huấn luyện nhanh mô hình xgboost và randomforest

from __future__ import annotations

from typing import Optional

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