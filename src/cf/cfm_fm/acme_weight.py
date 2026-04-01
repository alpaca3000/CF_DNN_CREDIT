# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

from __future__ import annotations

from typing import Any, Callable, List, Sequence

import numpy as np
import pandas as pd


class AcMEEngine:
    """
    Accelerated Model-agnostic Explanations (AcME) engine.
    Đã được tối ưu hóa Batch Prediction.
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
            baseline_dict[col] = df[col].mode()[0]
            
        baseline_series = pd.Series(baseline_dict)[num_features + cat_features]
        
        return cls(
            predict_proba=predict_proba,
            baseline_series=baseline_series,
            num_features=num_features,
            cat_features=cat_features,
            target_class=target_class,
        )

    def _predict_batch(self, x_df: pd.DataFrame) -> np.ndarray:
        """Thực hiện dự đoán theo Batch thay vì từng dòng để tối ưu tốc độ."""
        preds = self.predict_proba(x_df)
        arr = np.asarray(preds)
        if arr.ndim == 1:
            return arr
        return arr[:, self.target_class]

    def build_variation_matrix(
        self,
        df_train: pd.DataFrame,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    ) -> pd.DataFrame:
        """Tạo ma trận biến thể cho cả biến số và biến phân loại."""
        rows = []
        
        if self.num_features:
            q_df = df_train[self.num_features].quantile(quantiles, axis=0)
            for col in self.num_features:
                for q in quantiles:
                    row = self.baseline_series.copy()
                    row[col] = float(q_df.loc[q, col])
                    row_dict = {"feature": col, "variation": f"Q_{q}"}
                    row_dict.update(row.to_dict())
                    rows.append(row_dict)

        for col in self.cat_features:
            unique_vals = df_train[col].dropna().unique()
            for val in unique_vals:
                if val == self.baseline_series[col]:
                    continue
                row = self.baseline_series.copy()
                row[col] = val
                row_dict = {"feature": col, "variation": f"Cat_{val}"}
                row_dict.update(row.to_dict())
                rows.append(row_dict)

        return pd.DataFrame(rows)

    def compute_standardize_effect(
        self,
        variation_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """Tính toán Effect bằng Batch Prediction (Nhanh hơn rất nhiều)."""
        # 1. Tính xác suất cho Baseline
        baseline_df = pd.DataFrame([self.baseline_series])
        y_baseline = self._predict_batch(baseline_df)[0]
        
        # 2. Tính xác suất cho toàn bộ Variation Matrix cùng 1 lúc
        X_batch = variation_matrix[self.feature_names]
        y_hats = self._predict_batch(X_batch)
        
        # 3. Tổng hợp kết quả
        effects_df = variation_matrix[["feature", "variation"]].copy()
        effects_df["y_hat"] = y_hats
        effects_df["y_baseline"] = y_baseline
        effects_df["effect"] = y_hats - y_baseline
        
        return effects_df

    def explain(
        self,
        df_train: pd.DataFrame,
        quantiles: Sequence[float] = (0.25, 0.50, 0.75),
    ) -> dict[str, pd.DataFrame | pd.Series]:
        
        # Chỉ gọi build_variation_matrix 1 lần
        variation_matrix = self.build_variation_matrix(df_train, quantiles)
        effect_df = self.compute_standardize_effect(variation_matrix)
        
        # LỖI CŨ ĐÃ ĐƯỢC FIX: Lấy Abs() TRƯỚC khi tính Mean()
        abs_weights = effect_df.copy()
        abs_weights["abs_effect"] = abs_weights["effect"].abs()
        final_weights = abs_weights.groupby("feature")["abs_effect"].mean()
        
        # Điền 0 cho những feature không có sự thay đổi
        final_weights = final_weights.reindex(self.feature_names).fillna(0.0)

        # Chuẩn hóa trọng số (tổng = 1)
        weight_sum = final_weights.sum()
        if weight_sum > 0:
            final_weights = final_weights / weight_sum

        return {
            "variation_matrix": variation_matrix,
            "effects": effect_df,
            "weights": final_weights, # Trả về biến có tên ngắn gọn hơn
        }


if __name__ == "__main__":
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import sys
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.preprocess.preprocess import CreditPreprocessor

    print("=" * 70)
    print("[AcME TEST] German Credit - AcMEEngine sanity script")
    print("=" * 70)

    data_path = PROJECT_ROOT / "data" / "german_credit.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    df = pd.read_csv(data_path)
    if "Class" not in df.columns:
        raise ValueError("German Credit cần có cột nhãn 'Class'.")

    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = CreditPreprocessor(dataset_name="german", model_type="embedding")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train một RF model đơn giản để có hàm predict_proba
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_processed, y_train)

    # Tạo wrapper predict_proba cho DataFrame
    def predict_proba_wrapper(df_or_arr: Any) -> Any:
        if isinstance(df_or_arr, pd.DataFrame):
            arr = preprocessor.transform(df_or_arr)
        else:
            arr = df_or_arr
        return rf_model.predict_proba(arr)

    metadata = preprocessor.get_metadata()
    num_features = metadata["num_features"]
    cat_features = metadata["cat_features"]

    acme = AcMEEngine.from_dataframe_baseline(
        predict_proba=predict_proba_wrapper,
        df=X_train,
        num_features=num_features,
        cat_features=cat_features,
        target_class=1,
    )

    print(f"[OK] AcMEEngine initialized. Features: {len(acme.feature_names)}")
    print(f"     Num: {len(num_features)}, Cat: {len(cat_features)}")

    explanation = acme.explain(X_train, quantiles=(0.25, 0.5, 0.75))

    weights = explanation["weights"]
    print(f"[OK] weights shape: {weights.shape}")
    sorted_weights = weights.sort_values(ascending=False).iloc[:5]
    print(f"     top 5 features: {sorted_weights.index.tolist()}")
    print(f"     top 5 values: {sorted_weights.values.tolist()}")
    weight_sum_val = float(np.array(weights).sum())
    print(f"     weight sum: {weight_sum_val:.6f}")

    assert weights.shape[0] == len(acme.feature_names)
    assert np.isclose(weight_sum_val, 1.0, atol=1e-6) or weight_sum_val == 0.0
    assert (weights >= 0).all()

    effects = explanation["effects"]
    print(f"[OK] effects shape: {effects.shape}")
    print(f"     effects columns: {effects.columns.tolist()}")

    variation_matrix = explanation["variation_matrix"]
    print(f"[OK] variation_matrix shape: {variation_matrix.shape}")

    print("\n[SUCCESS] AcME test script chạy ổn trên German Credit.")
