# usage: python -m src.cf.cfm_fm.lof
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import warnings

class PlausibilityLOF:
    """
    Module quản lý tính hợp lý (Plausibility) cho Counterfactual Framework.
    Sử dụng thuật toán Local Outlier Factor (LOF) để đánh giá xem một 
    mẫu phản thực tế có nằm trong phân phối dữ liệu thực tế hay không.
    """
    
    def __init__(self, n_neighbors: int = 20, **kwargs):
        """
        Khởi tạo mô hình LOF.
        
        Args:
            n_neighbors (int): Số lượng láng giềng gần nhất để tính mật độ. 
                               Thường chọn từ 20 (mặc định của sklearn).
            **kwargs: Các tham số khác truyền vào LocalOutlierFactor.
        """
        # novelty=True là BẮT BUỘC để có thể dùng score_samples() và predict() trên dữ liệu mới
        self.lof_model = LocalOutlierFactor(
            n_neighbors=n_neighbors, 
            novelty=True, 
            **kwargs
        )
        
        self.min_train_lof = None
        self.max_train_lof = None
        self.is_fitted = False

    def fit(self, X_train_scaled: np.ndarray):
        """
        Huấn luyện mô hình LOF trên tập dữ liệu Train ĐÃ QUA TIỀN XỬ LÝ (Encoded & Scaled).
        Đồng thời tính toán và lưu lại điểm số min/max của tập Train để chuẩn hóa Min-Max sau này.
        
        Args:
            X_train_scaled (np.ndarray): Ma trận dữ liệu huấn luyện (chỉ chứa số).
        """
        if not isinstance(X_train_scaled, np.ndarray):
            X_train_scaled = np.array(X_train_scaled)

        # Huấn luyện mô hình
        self.lof_model.fit(X_train_scaled)
        self.is_fitted = True

        # Trích xuất điểm LOF của chính tập Train.
        # Lưu ý: Khi novelty=True, score_samples() trả về điểm âm (càng âm càng là outlier)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_scores = self.lof_model.score_samples(X_train_scaled)
        
        self.min_train_lof = float(np.min(train_scores))
        self.max_train_lof = float(np.max(train_scores))
        
        print(f"[PlausibilityLOF] Đã huấn luyện xong. Min LOF: {self.min_train_lof:.4f}, Max LOF: {self.max_train_lof:.4f}")

    def get_train_stats(self) -> dict:
        """
        Trả về dict chứa các thông số thống kê của tập Train.
        Dùng để truyền vào `CFMProblem` phục vụ hàm `plausibility_objective` (F4).
        """
        if not self.is_fitted:
            raise ValueError("Mô hình LOF chưa được huấn luyện. Vui lòng gọi hàm fit() trước.")
            
        return {
            'min': self.min_train_lof,
            'max': self.max_train_lof
        }

    def score_samples(self, X_new_scaled: np.ndarray) -> np.ndarray:
        """
        Đánh giá điểm LOF cho các mẫu phản thực tế mới.
        Dùng trong vòng lặp _evaluate của NSGA-II.
        
        Args:
            X_new_scaled (np.ndarray): Mẫu phản thực tế ĐÃ QUA TIỀN XỬ LÝ.
            
        Returns:
            np.ndarray: Mảng chứa điểm số LOF (giá trị âm).
        """
        if not self.is_fitted:
            raise ValueError("Mô hình LOF chưa được huấn luyện. Vui lòng gọi hàm fit() trước.")
            
        if not isinstance(X_new_scaled, np.ndarray):
            X_new_scaled = np.array(X_new_scaled)
            
        # Nếu truyền vào 1 sample dạng 1D (n_features,), reshape thành 2D (1, n_features)
        if X_new_scaled.ndim == 1:
            X_new_scaled = X_new_scaled.reshape(1, -1)
            
        return self.lof_model.score_samples(X_new_scaled)

    def predict_inliers(self, X_new_scaled: np.ndarray) -> np.ndarray:
        """
        Dự đoán nhãn Inlier/Outlier cho mẫu mới.
        Dùng cho bước đánh giá cuối cùng (tính metric LOF-NR trong metrics.py).
        
        Args:
            X_new_scaled (np.ndarray): Dữ liệu phản thực tế ĐÃ QUA TIỀN XỬ LÝ.
            
        Returns:
            np.ndarray: Nhãn dự đoán (1 là Inlier hợp lệ, -1 là Outlier dị biệt).
        """
        if not self.is_fitted:
            raise ValueError("Mô hình LOF chưa được huấn luyện. Vui lòng gọi hàm fit() trước.")
            
        if not isinstance(X_new_scaled, np.ndarray):
            X_new_scaled = np.array(X_new_scaled)
            
        if X_new_scaled.ndim == 1:
            X_new_scaled = X_new_scaled.reshape(1, -1)
            
        return self.lof_model.predict(X_new_scaled)


if __name__ == "__main__":
    from pathlib import Path
    import sys
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    from src.preprocess.preprocess import CreditPreprocessor

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    data_path = PROJECT_ROOT / "data" / "lending_club_balanced_sample_10k.csv"

    print("=" * 70)
    print("[LOF TEST] German Credit - PlausibilityLOF sanity script")
    print("=" * 70)

    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Lending Club cần có cột nhãn 'target'.")

    X = df.drop(columns=["target"]).copy()
    y = df["target"].copy()

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = CreditPreprocessor(dataset_name="lending_club", model_type="embedding")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    lof = PlausibilityLOF(n_neighbors=100)
    lof.fit(X_train_processed)

    stats = lof.get_train_stats()
    # score_samples của sklearn trả về điểm âm (càng âm càng bất thường).
    # Quy đổi về LOF "chuẩn" bằng công thức: lof_value = -score_samples,
    # khi đó mẫu bình thường thường gần 1, outlier thường > 1.
    print(f"[OK] Train stats: min={stats['min']:.6f}, max={stats['max']:.6f}")

    train_scores = lof.score_samples(X_train_processed)
    train_lof_values = -train_scores
    train_pct_leq_1 = (train_lof_values <= 1.0).mean() * 100
    print(f"[CHECK][TRAIN] % mẫu có LOF <= 1: {train_pct_leq_1:.2f}%")
    print(f"               % mẫu có LOF <= 1.2 : {(train_lof_values <= 1.2).mean() * 100:.2f}%")
    print(f"               % mẫu có LOF > 1 : {100 - train_pct_leq_1:.2f}%")

    test_scores = lof.score_samples(X_test_processed)
    test_lof_values = -test_scores
    test_pct_leq_1 = (test_lof_values <= 1.0).mean() * 100
    print(f"[OK] score_samples shape: {test_scores.shape}")
    print(f"     sample scores (5): {np.round(test_scores[:5], 6)}")
    print(f"     sample LOF (5): {np.round(test_lof_values[:5], 6)}")
    print(f"[CHECK][TEST ] % mẫu có LOF <= 1: {test_pct_leq_1:.2f}%")
    print(f"               % mẫu có LOF > 1 : {100 - test_pct_leq_1:.2f}%")

    pred_labels = lof.predict_inliers(X_test_processed)
    unique_labels = np.unique(pred_labels)
    print(f"[OK] predict labels unique: {unique_labels.tolist()}")
    print(f"     inlier ratio: {(pred_labels == 1).mean():.4f}")
    print(f"     outlier ratio: {(pred_labels == -1).mean():.4f}")

    assert np.isfinite(stats["min"]) and np.isfinite(stats["max"])
    assert test_scores.ndim == 1 and len(test_scores) == len(X_test_processed)
    assert set(unique_labels).issubset({-1, 1})

    one_score = lof.score_samples(X_test_processed[0])
    one_pred = lof.predict_inliers(X_test_processed[0])
    assert one_score.shape == (1,)
    assert one_pred.shape == (1,)
    assert np.isfinite(train_lof_values).all() and np.isfinite(test_lof_values).all()

    print("\n[SUCCESS] LOF test script chạy ổn trên German Credit.")