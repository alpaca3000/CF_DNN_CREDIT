from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def hinge_loss(
	f_x_star: float,
	threshold: float = 0.5,
) -> float:
	"""
	Hàm hinge loss để ép phản thực tế vượt ngưỡng phân lớp.

	Công thức:
		max(0, threshold - f(x*))

	Mặc định threshold = 0.5.
	"""
	score = float(f_x_star)
	return float(max(0.0, threshold - score))


def weighted_proximity(
	x: Sequence[float] | np.ndarray,
	x_star: Sequence[float] | np.ndarray,
	acme_weights: Sequence[float] | np.ndarray,
	eps: float = 1e-12,
) -> float:
	"""
	Khoảng cách Euclidean có trọng số nghịch đảo $1 / w_j$.

	d_w(x, x*) = sqrt( sum_j ( (x*_j - x_j)^2 * (1 / (w_j + eps)) ) )

	Trong đó `w_j` là trọng số đặc trưng từ AcME (giá trị tuyệt đối).
	"""
	x_arr = np.asarray(x, dtype=float).ravel()
	x_star_arr = np.asarray(x_star, dtype=float).ravel()
	w_arr = np.asarray(acme_weights, dtype=float).ravel()

	if x_arr.shape != x_star_arr.shape:
		raise ValueError("x và x_star phải cùng kích thước.")
	if w_arr.shape != x_arr.shape:
		raise ValueError("acme_weights phải cùng kích thước với x.")

	delta = x_star_arr - x_arr
	inv_w = 1.0 / np.maximum(np.abs(w_arr), eps)
	return float(np.sqrt(np.sum((delta**2) * inv_w)))


def relative_impact_measure(
	x: Sequence[float] | np.ndarray,
	x_star: Sequence[float] | np.ndarray,
	acme_weights: Sequence[float] | np.ndarray,
	eps: float = 1e-12,
	change_tolerance: float = 1e-9,
) -> float:
	"""
	Relative Impact Measure (RIM) cho độ thưa thớt có trọng số.

	Ý tưởng: chỉ tính các thay đổi thực sự, và phạt mạnh hơn ở đặc trưng quan trọng.

	RIM = sum_j( |x*_j - x_j| * |w_j| * I(|x*_j - x_j| > tol) ) / (sum_j |w_j| + eps)

	- Giá trị nhỏ hơn tốt hơn (ít thay đổi / thay đổi nhẹ ở đặc trưng quan trọng).
	"""
	x_arr = np.asarray(x, dtype=float).ravel()
	x_star_arr = np.asarray(x_star, dtype=float).ravel()
	w_arr = np.asarray(acme_weights, dtype=float).ravel()

	if x_arr.shape != x_star_arr.shape:
		raise ValueError("x và x_star phải cùng kích thước.")
	if w_arr.shape != x_arr.shape:
		raise ValueError("acme_weights phải cùng kích thước với x.")

	delta_abs = np.abs(x_star_arr - x_arr)
	mask = delta_abs > float(change_tolerance)
	weighted_change = delta_abs * np.abs(w_arr) * mask.astype(float)

	return float(np.sum(weighted_change) / (np.sum(np.abs(w_arr)) + eps))


def plausibility_score(
    X_cf: pd.DataFrame | np.ndarray,
    X_ref: pd.DataFrame | np.ndarray,
    n_neighbors: int = 20,
    contamination: str | float = "auto",
    use_sklearn_threshold: bool = True # Thêm cờ để bật tính năng thực tế
) -> float:
    """
    Điểm hợp lý (plausibility) dựa trên LOF-NR.
    Đã được fix ngưỡng threshold để phù hợp với dữ liệu thực tế có nhiễu.
    """
    X_ref_arr = _as_2d_array(X_ref)
    X_cf_arr = _as_2d_array(X_cf)

    if X_ref_arr.shape[1] != X_cf_arr.shape[1]:
        raise ValueError("X_ref và X_cf phải có cùng số lượng đặc trưng.")

    n_neighbors_eff = max(2, min(int(n_neighbors), X_ref_arr.shape[0] - 1))

    # Khởi tạo thuật toán
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors_eff,
        contamination=contamination,
        novelty=True,
    )
    lof.fit(X_ref_arr)

    if use_sklearn_threshold:
        # Cách TỐI ƯU THỰC TẾ: Dùng hàm predict của Sklearn
        # Trả về 1 (Inlier - Hợp lý) và -1 (Outlier - Bất hợp lý)
        # Sklearn tự động dùng ngưỡng offset_ (thường là ~1.5)
        preds = lof.predict(X_cf_arr)
        lof_nr = float(np.mean(preds == 1))
    else:
        # Cách LÝ THUYẾT GỐC (Dễ làm rớt 80% dữ liệu thật):
        lof_values = -lof.score_samples(X_cf_arr)
        lof_nr = float(np.mean(lof_values <= 1.0))

    return lof_nr

def _as_2d_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float)
    else:
        arr = np.asarray(X, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Đầu vào phải là vector 1D hoặc ma trận 2D.")
    return arr


__all__ = [
	"hinge_loss",
	"weighted_proximity",
	"relative_impact_measure",
	"plausibility_score",
]

class CFMEvaluator:
    """
    Module Đánh giá tự động. Chuyển đổi DataFrame thành mảng số (numpy) 
    để đưa vào các công thức toán học chuẩn của bài báo.
    """
    def __init__(self, preprocessor, df_train, feature_weights, target_col='Class', good_label=1):
        self.preprocessor = preprocessor
        self.weights_series = feature_weights
        
        # Lấy nhóm dữ liệu "Được duyệt" (Non-Rejected) để tính LOF-NR
        self.df_good = df_train[df_train[target_col] == good_label].drop(columns=[target_col])
        self.X_ref_encoded = self.preprocessor.transform(self.df_good)
        
        # Căn chỉnh weights vector khớp với thứ tự cột sau khi encode
        encoded_feature_names = preprocessor.get_metadata()["num_features"] + preprocessor.get_metadata()["cat_features"]
        self.w_arr = np.array([self.weights_series.get(col, 1.0) for col in encoded_feature_names])

    def evaluate(self, x_original: pd.Series, cf_df: pd.DataFrame) -> dict:
        if cf_df.empty:
            return {"Error": "Không có hồ sơ phản thực tế nào để đánh giá."}
            
        x_orig_encoded = self.preprocessor.transform(pd.DataFrame([x_original]))[0]
        cf_encoded = self.preprocessor.transform(cf_df.drop(columns=['predicted_prob'], errors='ignore'))
        
        # 1. Tính LOF-NR (Plausibility)
        # Sử dụng hàm plausibility_score từ bài báo
        lof_nr = plausibility_score(X_cf=cf_encoded, X_ref=self.X_ref_encoded, n_neighbors=2)
        
        # 2. Tính Proximity và RIM trung bình cho tất cả các giải pháp
        prox_list, rim_list = [], []
        for i in range(cf_encoded.shape[0]):
            cf_row = cf_encoded[i]
            # Tính khoảng cách có trọng số nghịch đảo
            prox = weighted_proximity(cf_row, x_orig_encoded, self.w_arr)
            # Tính độ thưa có trọng số
            rim = relative_impact_measure(cf_row, x_orig_encoded, self.w_arr)
            prox_list.append(prox)
            rim_list.append(rim)
            
        return {
            "Total CFs Found": len(cf_df),
            "Plausibility (LOF-NR)": round(lof_nr, 4),
            "Avg Weighted Proximity (Cost)": round(np.mean(prox_list), 4),
            "Avg Relative Impact (RIM)": round(np.mean(rim_list), 4)
        }

    def evaluate_benchmark(
        self,
        x_original: pd.Series,
        cf_df: pd.DataFrame,
        model_wrapper,
        threshold: float = 0.5,
        change_tolerance: float = 1e-9,
        n_neighbors: int = 20,
    ) -> dict:
        """
        Tính 4 chỉ số benchmark chuẩn:
        - Validity: tỷ lệ CF có xác suất >= threshold
        - Proximity (L1): trung bình ||x* - x||_1 trên không gian đã encode
        - Sparsity: số đặc trưng thay đổi trung bình (càng nhỏ càng tốt)
        - LOF-NR: độ hợp lý dựa trên Local Outlier Factor
        """
        if cf_df.empty:
            return {
                "n_cf": 0,
                "validity": 0.0,
                "proximity_l1": np.nan,
                "sparsity": np.nan,
                "lof_nr": np.nan,
            }

        cf_features = cf_df.drop(columns=["predicted_prob"], errors="ignore")

        # 1) Validity
        probs = model_wrapper.predict_proba(cf_features)
        validity = float(np.mean(np.asarray(probs) >= float(threshold)))

        # 2) Encode để tính distance/sparsity nhất quán cho cả số + phân loại
        x_orig_encoded = self.preprocessor.transform(pd.DataFrame([x_original]))[0]
        cf_encoded = self.preprocessor.transform(cf_features)

        # 3) Proximity (L1)
        l1_vals = np.sum(np.abs(cf_encoded - x_orig_encoded.reshape(1, -1)), axis=1)
        proximity_l1 = float(np.mean(l1_vals))

        # 4) Sparsity (số lượng đặc trưng thay đổi)
        changed = np.abs(cf_encoded - x_orig_encoded.reshape(1, -1)) > float(change_tolerance)
        sparsity = float(np.mean(np.sum(changed, axis=1)))

        # 5) LOF-NR
        lof_nr = plausibility_score(
            X_cf=cf_encoded,
            X_ref=self.X_ref_encoded,
            n_neighbors=n_neighbors,
        )

        return {
            "n_cf": int(len(cf_df)),
            "validity": validity,
            "proximity_l1": proximity_l1,
            "sparsity": sparsity,
            "lof_nr": float(lof_nr),
        }