# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

from __future__ import annotations

from typing import Any, Callable, List, Sequence

import numpy as np
import pandas as pd


class AcMEEngine:
    """
    Accelerated Model-agnostic Explanations (AcME) engine.

    Parameters
    ----------
    predict_proba : Callable
        Hàm của mô hình trả về xác suất dự đoán (giống API `predict_proba`).
    baseline_vector : Sequence[float]
        Véc-tơ baseline (thường là trung bình của tập dữ liệu).
    feature_names : Iterable[str] | None
        Danh sách tên đặc trưng. Nếu None sẽ tự sinh f0, f1, ...
    target_class : int
        Chỉ số lớp cần giải thích khi `predict_proba` trả về ma trận xác suất.
    """
    def __init__(
        self,
        predict_proba: Callable[[Any], Any],
        baseline_series: pd.Series,
        num_features: List[str],
        cat_features: List[str],
        target_class: int = 1,
    ) -> None:
        self.predict_proba = predict_proba
        self.baseline_series = baseline_series.copy()
        self.num_features = num_features
        self.cat_features = cat_features
        self.feature_names = num_features + cat_features
        self.n_features = len(self.feature_names)
        self.target_class = target_class

    @classmethod
    def from_dataframe_baseline(
        cls,
        predict_proba: Callable[[Any], Any],
        df: pd.DataFrame,
        num_features: List[str],
        cat_features: List[str],
        target_class: int = 1,
    ) -> "AcMEEngine":
        """Khởi tạo baseline: Mean cho biến số, Mode cho biến phân loại."""
        baseline_dict = {}
        for col in num_features:
            baseline_dict[col] = df[col].mean()
        for col in cat_features:
            baseline_dict[col] = df[col].mode()[0]  # Lấy giá trị phổ biến nhất
            
        baseline_series = pd.Series(baseline_dict)[num_features + cat_features]
        
        return cls(
            predict_proba=predict_proba,
            baseline_series=baseline_series,
            num_features=num_features,
            cat_features=cat_features,
            target_class=target_class,
        )

    def _predict_single(self, x_series: pd.Series) -> float:
        # Chuyển pd.Series thành DataFrame 1 dòng để đưa vào Wrapper
        x_df = pd.DataFrame([x_series])
        pred = self.predict_proba(x_df)
        
        arr = np.asarray(pred)
        if arr.ndim == 1:
            return float(arr[0]) if arr.shape[0] == 1 else float(arr[self.target_class])
        return float(arr[0, self.target_class])

    def build_variation_matrix(
        self,
        df_train: pd.DataFrame,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    ) -> pd.DataFrame:
        """Tạo ma trận biến thể cho cả biến số và biến phân loại."""
        rows = []
        
        # 1. Biến số -> Dùng Quantiles
        if self.num_features:
            q_df = df_train[self.num_features].quantile(quantiles, axis=0)
            for col in self.num_features:
                for q in quantiles:
                    row = self.baseline_series.copy()
                    row[col] = float(q_df.loc[q, col])
                    row_dict = {"feature": col, "variation": f"Q_{q}"}
                    row_dict.update(row.to_dict())
                    rows.append(row_dict)

        # 2. Biến phân loại -> Dùng Unique Categories
        for col in self.cat_features:
            unique_vals = df_train[col].dropna().unique()
            for val in unique_vals:
                if val == self.baseline_series[col]:
                    continue # Bỏ qua nếu giống hệt baseline
                row = self.baseline_series.copy()
                row[col] = val
                row_dict = {"feature": col, "variation": f"Cat_{val}"}
                row_dict.update(row.to_dict())
                rows.append(row_dict)

        return pd.DataFrame(rows)

    def compute_standardize_effect(
        self,
        df_train: pd.DataFrame,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    ) -> pd.DataFrame:
        variation_matrix = self.build_variation_matrix(df_train, quantiles)
        y_baseline = self._predict_single(self.baseline_series)

        effects = []
        for _, row in variation_matrix.iterrows():
            x_mod = row[self.feature_names]
            y_hat = self._predict_single(x_mod)
            effect = y_hat - y_baseline
            effects.append({
                "feature": row["feature"],
                "variation": row["variation"],
                "y_hat": y_hat,
                "y_baseline": y_baseline,
                "effect": effect,
            })

        return pd.DataFrame(effects)

    def explain(
        self,
        df_train: pd.DataFrame,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    ) -> dict[str, pd.DataFrame | pd.Series]:
        variation_matrix = self.build_variation_matrix(df_train, quantiles)
        effect_df = self.compute_standardize_effect(df_train, quantiles)
        
        abs_weights = effect_df.groupby("feature")["effect"].mean().abs()
        # Điền 0 cho những feature không có sự thay đổi
        abs_weights = abs_weights.reindex(self.feature_names).fillna(0.0)

        # Chuẩn hóa trọng số (tổng = 1) để tiện dùng cho hàm mục tiêu NSGA-II sau này
        if abs_weights.sum() > 0:
            abs_weights = abs_weights / abs_weights.sum()

        return {
            "variation_matrix": variation_matrix,
            "effects": effect_df,
            "abs_weights": abs_weights,
        }
