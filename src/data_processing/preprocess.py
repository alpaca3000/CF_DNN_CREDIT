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
    """

    def __init__(self, dataset_name: str = "german_credit", model_type: str = "embedding") -> None:
        self.dataset_name = dataset_name
        self.model_type = model_type

        self.scaler: Optional[StandardScaler] = None
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.onehot_encoder: Optional[OneHotEncoder] = None

        self.fitted_ = False
        self.num_features_: List[str] = []
        self.cat_features_: List[str] = []
        self.num_medians_: Dict[str, float] = {}
        self.num_ranges_: Dict[str, float] = {}

        self.configs: Dict[str, Dict[str, Any]] = {
            "german_credit": {
                # Chú ý: Đưa "Age", "Employment" sang Actionable để áp dụng được luật >=
                "actionable": [
                    "Amount", "Duration", "InstallmentRate", 
                    "OtherDebtors", "OtherPlans", "Telephone", 
                    "ExistingCredits", "Status", "Savings", "Property"
                ],
                
                # Nhóm 1 & 2: Khóa chặt để tránh đưa ra lời khuyên phi thực tế hoặc phi đạo đức
                "immutable": [
                    "PersonalStatus", "ForeignWorker", "History", "Liable", "Purpose", 
                    "Age", "Employment", "ResidenceSince", "Job", "Housing"
                ],
                
                # Ép logic 1 chiều cho các biến Actionable
                "causal_rules": [
                    {"feature": "Amount", "type": "<="},             # Khuyên giảm số tiền vay
                    # {"feature": "ExistingCredits", "type": "<="},  # Khuyên trả bớt nợ cũ
                    # {"feature": "Status", "type": ">="},           # Cải thiện số dư tài khoản
                    # {"feature": "Savings", "type": ">="},          # Tăng tiền tiết kiệm
                    {"feature": "Duration", "type": "<="},           # Khuyên giảm thời gian vay
                    {"feature": "InstallmentRate", "type": ">="},    # Khuyên tăng lệ trả góp
                    {"feature": "ExistingCredits", "type": "<="},    # Khuyên giảm số khoản vay mở hiện tại 
                    # cho tất cả các biến immutable là không đổi
                    {"feature": "PersonalStatus", "type": "=="},
                    {"feature": "ForeignWorker", "type": "=="},
                    {"feature": "History", "type": "=="},
                    {"feature": "Liable", "type": "=="},
                    {"feature": "Purpose", "type": "=="},
                    {"feature": "Age", "type": "=="},
                    {"feature": "Employment", "type": "=="},
                    {"feature": "ResidenceSince", "type": "=="},
                    {"feature": "Job", "type": "=="},
                    {"feature": "Housing", "type": "=="}
                ],
            },
            "lending_club": {
                "actionable": [ 
                    "log_loan_amnt", "log_annual_inc", "dti", "revol_util", "inst_to_inc_ratio"
                ],
                "immutable": [ 
                    "term_num", "sub_grade", "home_ownership", "purpose_grouped", 
                    "credit_hist_months", "state_risk_score"
                ],
                "categorical": [
                    "sub_grade", "home_ownership", "purpose_grouped"
                ],
                "numerical": [
                    "log_loan_amnt", "term_num", "log_annual_inc", "dti", 
                    "revol_util", "log_revol_bal", "credit_hist_months", 
                    "inst_to_inc_ratio", "state_risk_score", "emp_length"
                ],
                "target": "target"
            },
            "gmsc": {
                "actionable": [ # Các biến có thể tác động (Mutable)
                    "RevolvingUtilizationOfUnsecuredLines", 
                    "DebtRatio", 
                    "MonthlyIncome", 
                    "NumberOfOpenCreditLinesAndLoans"
                ],
                "immutable": [ # Các biến cố định (Immutable)
                    "age", 
                    "NumberOfTime30-59DaysPastDueNotWorse",
                    "NumberOfTimes90DaysLate", 
                    "NumberRealEstateLoansOrLines", 
                    "NumberOfTime60-89DaysPastDueNotWorse", 
                    "NumberOfDependents"
                ],
                "categorical": [
                    "NumberOfDependents" # Giữ biến này là categorical để dùng Embedding
                ],
                "numerical": [
                    "RevolvingUtilizationOfUnsecuredLines", "age", 
                    "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", 
                    "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", 
                    "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines", 
                    "NumberOfTime60-89DaysPastDueNotWorse"
                ],
                "target": "SeriousDlqin2yrs"
            },
        }

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.copy()
    
    def _infer_feature_types(self, X: pd.DataFrame) -> tuple[List[str], List[str]]:
        # 1. Kiểm tra cấu hình có sẵn cho dataset này không
        cfg = self.configs.get(self.dataset_name, {})
        cfg_cat = cfg.get("categorical", [])
        cfg_num = cfg.get("numerical", [])

        if cfg_cat or cfg_num:
            # Nếu có cấu hình, ưu tiên lấy các cột đó (nếu chúng tồn tại trong X)
            cat_cols = [c for c in cfg_cat if c in X.columns]
            # Các cột numerical là những cột còn lại
            num_cols = [c for c in X.columns if c not in cat_cols]
            return num_cols, cat_cols
        
        # 2. Nếu không có cấu hình, tự động suy luận theo kiểu dữ liệu
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

        for col in self.num_features_:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            range_val = max_val - min_val
            # Đảm bảo không bị chia cho 0 nếu biến là hằng số
            self.num_ranges_[col] = range_val if range_val > 0 else 1e-9

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
        causal_rules = cfg.get("causal_rules", [])

        return {
            "num_features": self.num_features_,
            "cat_features": self.cat_features_,
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
            "actionable": actionable,
            "immutable": immutable,
            "causal_rules": causal_rules,
            "num_ranges_dict": self.num_ranges_,
        }

    def inverse_transform(self, X_transformed: np.ndarray) -> pd.DataFrame:
        """
        Đưa dữ liệu từ không gian model (scaled/encoded) về dạng raw DataFrame.

        Hỗ trợ:
        - embedding/tree: [num_scaled, cat_ordinal]
        - classic_mlp: [num_scaled, cat_onehot]
        """
        if not self.fitted_:
            raise RuntimeError("CreditPreprocessor chưa fit. Hãy gọi fit(X_train) trước.")

        X_arr = np.asarray(X_transformed)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        n_num = len(self.num_features_)
        n_cat = len(self.cat_features_)

        # Tách numeric
        if n_num > 0:
            num_part = X_arr[:, :n_num].astype(np.float32)
            if self.scaler is not None:
                num_raw = self.scaler.inverse_transform(num_part)
            else:
                num_raw = num_part
        else:
            num_raw = np.empty((X_arr.shape[0], 0), dtype=np.float32)

        # Tách categorical
        if n_cat > 0:
            cat_part = X_arr[:, n_num:]

            if self.model_type == "classic_mlp" and self.onehot_encoder is not None:
                # onehot -> category string
                cat_raw = self.onehot_encoder.inverse_transform(cat_part)
            else:
                # ordinal -> category string
                if self.ordinal_encoder is None:
                    raise RuntimeError("ordinal_encoder chưa được khởi tạo để inverse_transform.")

                cat_ord = np.rint(cat_part).astype(np.int64)
                # clip vào miền hợp lệ cho từng cột
                for j, cats in enumerate(self.ordinal_encoder.categories_):
                    max_idx = len(cats) - 1
                    cat_ord[:, j] = np.clip(cat_ord[:, j], 0, max_idx)

                cat_raw = self.ordinal_encoder.inverse_transform(cat_ord)
        else:
            cat_raw = np.empty((X_arr.shape[0], 0), dtype=object)

        out = pd.DataFrame(
            np.hstack([num_raw, cat_raw]),
            columns=self.num_features_ + self.cat_features_
        )

        # Ép lại kiểu số cho cột numeric
        for col in self.num_features_:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        return out
