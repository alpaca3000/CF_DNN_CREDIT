import numpy as np
import pandas as pd
from typing import Any

class CFMEvaluator:
    """
    Module Đánh giá tự động. Đo lường 6 chỉ số Benchmark:
    Validity, Cont-Proximity, Cont-Sparsity, Cat-Sparsity, LOF-NR.
    """
    def __init__(
        self, 
        preprocessor: Any, 
        plausibility_module: Any, 
        df_train_raw: pd.DataFrame
    ):
        """
        Khởi tạo Evaluator với các "tri thức" đã được huấn luyện từ trước.
        """
        self.preprocessor = preprocessor
        self.metadata = preprocessor.get_metadata()
        self.num_features = self.metadata["num_features"]
        self.cat_features = self.metadata["cat_features"]
        
        # Sử dụng module LOF đã được train từ Generator/Offline Setup
        self.plausibility_module = plausibility_module
        
        # Tính toán ranges (Max - Min) để chuẩn hóa Cont-Proximity
        ranges = []
        for col in self.num_features:
            col_range = float(df_train_raw[col].max() - df_train_raw[col].min())
            ranges.append(col_range if col_range > 0 else 1e-9)
        self.num_ranges = np.array(ranges)

    def evaluate(self, x_original: pd.Series, cf_df: pd.DataFrame) -> dict:
        """
        Đánh giá một tập hợp các mẫu CF so với mẫu gốc.
        cf_df PHẢI CHỨA dữ liệu dạng thô (giống với x_original).
        """
        if cf_df is None or cf_df.empty:
            return self._empty_result()

        cf_features = cf_df.drop(columns=["predicted_prob"], errors="ignore")
        num_cfs = len(cf_df)
        
        # 0. VALIDITY
        if "predicted_prob" in cf_df.columns:
            validity = float(np.mean(cf_df["predicted_prob"] >= 0.50))
        else:
            validity = 1.0 # Giả định các mẫu truyền vào đều đã hợp lệ

        # --- CHUẨN BỊ DỮ LIỆU ---
        x_orig_num = pd.to_numeric(x_original[self.num_features]).values
        cf_num = cf_features[self.num_features].apply(pd.to_numeric).values
        
        x_orig_cat = x_original[self.cat_features].values
        cf_cat = cf_features[self.cat_features].values
        
        # 1. CONTINUOUS METRICS (Sparsity & Proximity)
        if len(self.num_features) > 0:
            # Sparsity: TỶ LỆ đặc trưng KHÔNG bị thay đổi
            # = số đặc trưng giữ nguyên / tổng số đặc trưng liên tục
            diff_mask = ~np.isclose(cf_num, x_orig_num, atol=1e-5)
            unchanged_mask = ~diff_mask
            cont_spars = np.mean(np.sum(unchanged_mask, axis=1) / len(self.num_features))
            
            # Proximity: Khoảng cách L1 ĐÃ CHUẨN HÓA (chia cho ranges)
            # Điều này giúp Proximity không bị phụ thuộc vào đơn vị đo lường
            normalized_diff = np.abs(cf_num - x_orig_num) / self.num_ranges
            cont_prox = np.mean(np.sum(normalized_diff, axis=1))
        else:
            cont_prox, cont_spars = 0.0, 0.0

        # 2. CATEGORICAL METRICS (Sparsity & Proximity)
        if len(self.cat_features) > 0:
            cat_diff = (cf_cat != x_orig_cat).astype(int)
            # Sparsity: TỶ LỆ đặc trưng phân loại KHÔNG bị thay đổi
            # = số đặc trưng giữ nguyên / tổng số đặc trưng phân loại
            cat_unchanged = 1 - cat_diff
            cat_spars = np.mean(np.sum(cat_unchanged, axis=1) / len(self.cat_features))
        else:
            cat_spars = 0.0, 0.0

        # 3. LOF-NR (Novelty Rejection - Plausibility)
        # Sử dụng preprocessor hệ thống thay vì tự viết lại scaler/encoder
        cf_scaled = self.preprocessor.transform(cf_features)
        
        # Lấy điểm số từ mô hình (sklearn trả về số âm)
        negative_lof_scores = self.plausibility_module.score_samples(cf_scaled)
        
        # Đảo dấu để trở thành điểm LOF dương nguyên thủy (LOF >= 0)
        # Theo lý thuyết: LOF ~ 1 là inlier, LOF >> 1 là outlier
        lof_scores = -negative_lof_scores
        
        # Áp dụng TRỰC TIẾP bất đẳng thức của bài báo (Ví dụ ngưỡng là 1.1)
        inlier_threshold = 1.15 
        
        # Đếm tỷ lệ mẫu thỏa mãn bất đẳng thức: LOF <= ngưỡng
        lof_nr = float(np.mean(lof_scores <= inlier_threshold))

        return {
            "Validity": round(validity, 4),
            "Cont-Proximity": round(float(cont_prox), 4),
            "Cont-Sparsity": round(float(cont_spars), 4),
            "Cat-Sparsity": round(float(cat_spars), 4),
            "Plausibility (LOF-NR)": round(lof_nr, 4),
            "Total CFs Found": num_cfs
        }

    def _empty_result(self) -> dict:
        return {
            "Validity": 0.0,
            "Cont-Proximity": 0.0,
            "Cont-Sparsity": 0.0,
            "Cat-Sparsity": 0.0,
            "Plausibility (LOF-NR)": 0.0,
            "Total CFs Found": 0
        }