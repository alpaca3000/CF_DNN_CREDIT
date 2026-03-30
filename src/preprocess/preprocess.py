from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class CreditPreprocessor:
    """
    Leakage-safe preprocessor.

    - `fit()` chỉ dùng trên train split
    - `transform()` dùng cho val/test

    model_type:
      - `tree`: ordinal encode categorical, no scaling
      - `classic_mlp`: scale numeric + one-hot categorical
      - `embedding`: scale numeric + ordinal categorical
      - `tabnet`: scale numeric + ordinal categorical
    """

    def __init__(self, dataset_name: str = "german", model_type: str = "embedding") -> None:
        self.dataset_name = dataset_name
        self.model_type = model_type

        self.scaler: Optional[StandardScaler] = None
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.onehot_encoder: Optional[OneHotEncoder] = None

        self.fitted_ = False
        self.num_features_: List[str] = []
        self.cat_features_: List[str] = []
        self.num_medians_: Dict[str, float] = {}

        self.configs: Dict[str, Dict[str, Any]] = {
            "german": {
                # Đưa Nhóm 3 và 4 vào đây để NSGA-II tối ưu
                "actionable": [
                    "Amount", "Duration", "Purpose", "InstallmentRate", "OtherDebtors", 
                    "Status", "Savings", "Property", "OtherPlans", "ExistingCredits", "Telephone"
                ],
                # Đưa Nhóm 1 và Nhóm 2 vào đây. 
                # - Nhóm 1 sẽ bị khóa cứng ở create_pymoo_space.
                # - Nhóm 2 sẽ bị kiểm soát bởi mảng G trong CFMProblem.
                "immutable": [
                    "Age", "PersonalStatus", "ForeignWorker", "Liable", "Housing",
                    "Employment", "ResidenceSince", "Job", "History"
                ],
            },
            "lending_club": {
                "actionable": ["loan_amnt", "term", "annual_inc"],
                "immutable": ["emp_length", "addr_state"],
            },
            "gmsc": {
                "actionable": ["MonthlyIncome", "DebtRatio", "NumberOfOpenCreditLinesAndLoans"],
                "immutable": ["age"],
            },
        }

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.copy()

    @staticmethod
    def _infer_feature_types(X: pd.DataFrame) -> tuple[List[str], List[str]]:
        num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        return num_cols, cat_cols

    def _clean_features(self, X: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        X = X.copy()

        if not self.num_features_ and not self.cat_features_:
            self.num_features_, self.cat_features_ = self._infer_feature_types(X)

        for col in self.num_features_:
            if col not in X.columns:
                X[col] = 0.0
            if is_fit:
                med = pd.to_numeric(X[col], errors="coerce").median()
                self.num_medians_[col] = float(med) if pd.notna(med) else 0.0
            fill_value = self.num_medians_.get(col, 0.0)
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill_value)

        for col in self.cat_features_:
            if col not in X.columns:
                X[col] = "MISSING"
            X[col] = X[col].astype(str).fillna("MISSING")

        ordered_cols = self.num_features_ + self.cat_features_
        return X[ordered_cols]

    def fit(self, X_train: pd.DataFrame) -> "CreditPreprocessor":
        X = self._validate_input(X_train)
        self.num_features_, self.cat_features_ = self._infer_feature_types(X)
        X = self._clean_features(X, is_fit=True)

        if self.model_type in {"classic_mlp", "embedding", "tree"} and self.num_features_:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.num_features_])

        if self.model_type in {"embedding", "tree"} and self.cat_features_:
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.ordinal_encoder.fit(X[self.cat_features_].astype(str))

        if self.model_type == "classic_mlp" and self.cat_features_:
            try:
                self.onehot_encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
            except TypeError:
                self.onehot_encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                )
            self.onehot_encoder.fit(X[self.cat_features_].astype(str))

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("CreditPreprocessor chưa fit. Hãy gọi fit(X_train) trước.")

        X_df = self._validate_input(X)
        X_df = self._clean_features(X_df, is_fit=False)

        num_arr = (
            X_df[self.num_features_].to_numpy(dtype=np.float32)
            if self.num_features_
            else np.empty((len(X_df), 0), dtype=np.float32)
        )
        cat_arr_ord = np.empty((len(X_df), 0), dtype=np.int64)

        if self.num_features_ and self.scaler is not None:
            num_arr = self.scaler.transform(X_df[self.num_features_]).astype(np.float32)

        if self.cat_features_:
            cat_df = X_df[self.cat_features_].astype(str)
            if self.model_type == "classic_mlp" and self.onehot_encoder is not None:
                cat_onehot = self.onehot_encoder.transform(cat_df).astype(np.float32)
                return np.hstack([num_arr, cat_onehot]).astype(np.float32)

            if self.ordinal_encoder is not None:
                cat_arr_ord = self.ordinal_encoder.transform(cat_df)
                cat_arr_ord = np.where(cat_arr_ord < 0, 0, cat_arr_ord).astype(np.int64)

        if self.model_type == "tree":
            return np.hstack([num_arr.astype(np.float32), cat_arr_ord.astype(np.float32)]).astype(np.float32)

        return np.hstack([num_arr, cat_arr_ord.astype(np.float32)]).astype(np.float32)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def get_metadata(self) -> Dict[str, Any]:
        cat_idxs = list(range(len(self.num_features_), len(self.num_features_) + len(self.cat_features_)))
        cat_dims: List[int] = []

        if self.ordinal_encoder is not None and self.cat_features_:
            for cats in self.ordinal_encoder.categories_:
                cat_dims.append(int(len(cats)))

        cfg = self.configs.get(self.dataset_name, {})
        actionable = [c for c in cfg.get("actionable", []) if c in self.num_features_ + self.cat_features_]
        immutable = [c for c in cfg.get("immutable", []) if c in self.num_features_ + self.cat_features_]

        return {
            "num_features": self.num_features_,
            "cat_features": self.cat_features_,
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
            "actionable": actionable,
            "immutable": immutable,
        }