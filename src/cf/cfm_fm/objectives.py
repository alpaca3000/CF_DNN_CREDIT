# usage: python -m src.cf.cfm_fm.objectives
import numpy as np

# Loss function dùng để optimize trong quá trình tìm kiếm CF

# Validity: Hinge Loss (đảm bảo CF lật được nhãn)
def validity(f_x_star: float, threshold: float = 0.5) -> float:
    """
    F1: Tính toán Hinge Loss cho Validity.
    Mục tiêu: Đẩy xác suất dự đoán của lớp mục tiêu vượt qua ngưỡng threshold.
    Output: Nằm trong khoảng [0, 1]. Nếu đã vượt ngưỡng, trả về 0.0 (không bị phạt).
    """
    score = float(f_x_star)
    return float(max(0.0, threshold - score))

def proximity(
    x: np.ndarray, 
    x_star: np.ndarray, 
    weights: np.ndarray, 
    num_ranges: np.ndarray,
    cat_mask: np.ndarray = None, 
    eps: float = 1e-9
) -> float:
    """
    F2: Tính khoảng cách có trọng số (Weighted Proximity).
    Khuyến khích giữ lại các giá trị gốc, đặc biệt là các đặc trưng có trọng số cao.
    
    Args:
        x: Vector mẫu gốc (raw data).
        x_star: Vector mẫu phản thực tế (raw data).
        weights: Trọng số đặc trưng (ví dụ: từ AcME).
        num_ranges: Mảng chứa giá trị (Max - Min) của các biến liên tục trên tập train.
        cat_mask: Mảng boolean, True tại vị trí biến phân loại.
    """
    if cat_mask is None:
        cat_mask = np.zeros(len(x), dtype=bool)
        
    num_mask = ~cat_mask
    dist_sq = 0.0

    # 1. Biến liên tục: Chuẩn hóa khoảng cách bằng ranges, tính bình phương (L2 proxy) nhân trọng số
    if np.any(num_mask):
        x_num = x[num_mask].astype(float)
        x_star_num = x_star[num_mask].astype(float)
        w_num = np.abs(weights[num_mask].astype(float))
        ranges = num_ranges.astype(float) + eps
        
        # Chuẩn hóa khoảng cách về [0, 1] trước khi bình phương
        delta_norm = np.abs(x_num - x_star_num) / ranges
        dist_sq += np.sum((delta_norm ** 2) * w_num)

    # 2. Biến phân loại: Hamming distance nhân trọng số
    if np.any(cat_mask):
        x_cat = x[cat_mask]
        x_star_cat = x_star[cat_mask]
        w_cat = np.abs(weights[cat_mask].astype(float))
        
        diff = (x_cat != x_star_cat).astype(float)
        dist_sq += np.sum(diff * w_cat)

    # Chuẩn hóa tổng thể về [0, 1] bằng cách chia cho tổng trọng số
    total_weight = np.sum(np.abs(weights)) + eps
    normalized_dist = np.sqrt(dist_sq) / np.sqrt(total_weight)
    
    return float(np.clip(normalized_dist, 0.0, 1.0))

def rim(
    x: np.ndarray, 
    x_star: np.ndarray, 
    weights: np.ndarray, 
    num_ranges: np.ndarray,
    cat_mask: np.ndarray = None, 
    eps: float = 1e-9
) -> float:
    """
    F3: Relative Impact Measure (RIM).
    Đo lường sự bất hợp đồng do thay đổi thuộc tính gây ra, đánh giá mức độ 
    tác động tương đối lên mô hình thay vì chỉ đếm số lượng đặc trưng thay đổi (Sparsity L0).
    """
    if cat_mask is None:
        cat_mask = np.zeros(len(x), dtype=bool)
        
    num_mask = ~cat_mask
    total_impact = 0.0

    # 1. Tác động của biến liên tục (Dùng L1 norm đã chuẩn hóa)
    if np.any(num_mask):
        x_num = x[num_mask].astype(float)
        x_star_num = x_star[num_mask].astype(float)
        w_num = np.abs(weights[num_mask].astype(float))
        ranges = num_ranges.astype(float) + eps
        
        # Khoảng cách L1 chuẩn hóa
        delta_norm = np.abs(x_num - x_star_num) / ranges
        total_impact += np.sum(delta_norm * w_num)

    # 2. Tác động của biến phân loại
    if np.any(cat_mask):
        x_cat = x[cat_mask]
        x_star_cat = x_star[cat_mask]
        w_cat = np.abs(weights[cat_mask].astype(float))
        
        diff = (x_cat != x_star_cat).astype(float)
        total_impact += np.sum(diff * w_cat)

    # Chia cho tổng trọng số để ra tỷ lệ phần trăm tác động [0, 1]
    total_weight = np.sum(np.abs(weights)) + eps
    rim_score = total_impact / total_weight
    
    return float(np.clip(rim_score, 0.0, 1.0))

def plausibility(
    lof_score: float, 
    min_train_lof: float, 
    max_train_lof: float
) -> float:
    """
    F4: Plausibility (Tính hợp lý) thông qua Local Outlier Factor.
    Hàm score_samples của sklearn trả về giá trị âm (càng âm càng là outlier).
    Mục tiêu tối ưu là minimize, nên ta cần ánh xạ nó về khoảng [0, 1].
    Giá trị gần 1.0 nghĩa là mẫu rất dị thường (outlier nặng).
    """
    # Đảo dấu để đưa về bài toán minimize (điểm càng cao càng xấu)
    inverted_score = -lof_score
    inverted_min = -max_train_lof # Điểm inlier tốt nhất (âm ít nhất -> nhỏ nhất sau khi đảo)
    inverted_max = -min_train_lof # Điểm outlier tệ nhất (âm nhiều nhất -> lớn nhất sau khi đảo)
    
    # Min-Max Scaling
    range_lof = (inverted_max - inverted_min)
    if range_lof <= 0:
        return 0.0
        
    normalized_lof = (inverted_score - inverted_min) / range_lof
    return float(np.clip(normalized_lof, 0.0, 1.0))

# if __name__ == "__main__":
#     from pathlib import Path
#     import sys

#     import pandas as pd
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.model_selection import train_test_split

#     PROJECT_ROOT = Path(__file__).resolve().parents[3]
#     if str(PROJECT_ROOT) not in sys.path:
#         sys.path.insert(0, str(PROJECT_ROOT))

#     from src.data_processing.preprocess import CreditPreprocessor
#     from src.cf.cfm_fm.acme_weight import AcMEEngine
#     from src.cf.cfm_fm.lof import PlausibilityLOF

#     print("=" * 72)
#     print("[OBJECTIVES TEST] German Credit - validity/proximity/rim/plausibility")
#     print("=" * 72)

#     data_path = PROJECT_ROOT / "data" / "german_credit.csv"
#     if not data_path.exists():
#         raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

#     df = pd.read_csv(data_path)
#     if "Class" not in df.columns:
#         raise ValueError("German Credit phải có cột nhãn 'Class'.")

#     X = df.drop(columns=["Class"]).copy()
#     y = df["Class"].astype(int).copy()

#     X_train, X_test, y_train, _ = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y,
#     )

#     # 1) Preprocess + train model dự đoán xác suất
#     preprocessor = CreditPreprocessor(dataset_name="german", model_type="embedding")
#     X_train_proc = preprocessor.fit_transform(X_train)
#     X_test_proc = preprocessor.transform(X_test)

#     model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     model.fit(X_train_proc, y_train)

#     # 2) AcME để lấy trọng số đặc trưng
#     def predict_proba_wrapper(df_or_arr):
#         if isinstance(df_or_arr, pd.DataFrame):
#             arr = preprocessor.transform(df_or_arr)
#             return model.predict_proba(arr)
#         return model.predict_proba(df_or_arr)

#     metadata = preprocessor.get_metadata()
#     num_features = list(metadata["num_features"])
#     cat_features = list(metadata["cat_features"])
#     all_features = num_features + cat_features

#     acme = AcMEEngine.from_dataframe_baseline(
#         predict_proba=predict_proba_wrapper,
#         df=X_train,
#         num_features=num_features,
#         cat_features=cat_features,
#         target_class=1,
#     )
#     explanation = acme.explain(X_train)
#     weight_series = explanation["weights"]
#     w_arr = np.array([float(weight_series.get(col, 0.0)) for col in all_features], dtype=float)

#     # 3) LOF để test plausibility
#     lof = PlausibilityLOF(n_neighbors=20)
#     lof.fit(X_train_proc)
#     lof_stats = lof.get_train_stats()

#     # 4) Tạo 1 cặp x, x* đơn giản từ test
#     x_orig_series = X_test.iloc[0].copy()
#     x_cf_series = x_orig_series.copy()

#     # thay đổi 1 feature số (nếu có)
#     if len(num_features) > 0:
#         col_num = num_features[0]
#         span = float(X_train[col_num].max() - X_train[col_num].min())
#         step = 0.1 * span if span > 0 else 1.0
#         x_cf_series[col_num] = float(pd.to_numeric(x_cf_series[col_num], errors="coerce")) + step

#     # thay đổi 1 feature categorical (nếu có)
#     if len(cat_features) > 0:
#         col_cat = cat_features[0]
#         uniq_vals = X_train[col_cat].dropna().astype(str).unique().tolist()
#         current_val = str(x_cf_series[col_cat])
#         candidates = [v for v in uniq_vals if v != current_val]
#         if len(candidates) > 0:
#             x_cf_series[col_cat] = candidates[0]

#     x_orig = x_orig_series[all_features].to_numpy(dtype=object)
#     x_cf = x_cf_series[all_features].to_numpy(dtype=object)

#     cat_mask = np.array([col in cat_features for col in all_features], dtype=bool)
#     num_ranges = np.array(metadata.get("num_ranges", [1.0] * len(num_features)), dtype=float)

#     # 5) Tính 4 objective
#     prob_cf = float(model.predict_proba(preprocessor.transform(x_cf_series.to_frame().T))[0, 1])
#     f1 = validity(prob_cf, threshold=0.51)
#     f2 = proximity(x_orig, x_cf, w_arr, num_ranges, cat_mask=cat_mask)
#     f3 = rim(x_orig, x_cf, w_arr, num_ranges, cat_mask=cat_mask)

#     lof_score_cf = float(lof.score_samples(preprocessor.transform(x_cf_series.to_frame().T))[0])
#     f4 = plausibility(lof_score_cf, lof_stats["min"], lof_stats["max"])

#     print(f"prob_cf = {prob_cf:.6f}")
#     print(f"F1 validity      = {f1:.6f}")
#     print(f"F2 proximity     = {f2:.6f}")
#     print(f"F3 rim           = {f3:.6f}")
#     print(f"F4 plausibility  = {f4:.6f}")

#     for score, name in zip([f1, f2, f3, f4], ["F1", "F2", "F3", "F4"]):
#         assert 0.0 <= score <= 1.0, f"{name}={score} nằm ngoài [0,1]"

#     print("\n[SUCCESS] 4 objectives chạy đúng trên German Credit.")