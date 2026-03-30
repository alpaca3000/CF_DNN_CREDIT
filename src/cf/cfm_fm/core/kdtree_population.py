import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from pymoo.core.sampling import Sampling
from typing import List, Dict, Any
import sys
from pathlib import Path

# Allow running this file directly: `python testwraper.py`
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocess.preprocess import CreditPreprocessor

class KDTreeInitializer:
    """
    Tìm kiếm các hồ sơ láng giềng gần nhất đã được duyệt vay để làm "hạt giống" cho thuật toán.
    """
    def __init__(self, df_train: pd.DataFrame, preprocessor: Any, target_col: str, good_label: int = 1):
        # 1. Chỉ lọc ra những khách hàng ĐÃ ĐƯỢC DUYỆT VAY
        self.df_good = df_train[df_train[target_col] == good_label].copy()
        
        # 2. Xóa cột target để không đưa nhãn vào tính khoảng cách
        self.df_good_features = self.df_good.drop(columns=[target_col])
        self.preprocessor = preprocessor
        
        # 3. Sử dụng bộ Preprocessor để chuẩn hóa dữ liệu thành ma trận số
        X_encoded = self.preprocessor.transform(self.df_good_features)
        
        # 4. Huấn luyện KDTree trên không gian đã mã hóa
        self.kdtree = KDTree(X_encoded, metric='euclidean')

    def get_initial_population(self, x_original: pd.DataFrame, k: int = 50) -> List[Dict[str, Any]]:
        """
        Tìm K hồ sơ tốt gần giống x_original nhất.
        """
        # Encode hồ sơ khách hàng đang bị từ chối
        x_enc = self.preprocessor.transform(x_original)
        
        # Truy vấn KDTree lấy K láng giềng
        k_actual = min(k, len(self.df_good_features))
        distances, indices = self.kdtree.query(x_enc, k=k_actual)
        
        # Trích xuất các dòng dữ liệu gốc (dạng chữ/số nguyên thủy, chưa bị encode)
        nearest_records = self.df_good_features.iloc[indices[0]]
        
        # Chuyển về list các dictionary để pymoo sử dụng
        return nearest_records.to_dict(orient='records')


class KDTreeMixedSampling(Sampling):
    """
    Bộ bọc (Wrapper) đưa danh sách KDTree vào pymoo đúng chuẩn API Mixed Variable.
    Giải quyết triệt để lỗi "TypeError: 'list' object is not callable".
    """
    def __init__(self, kdtree_results: List[Dict[str, Any]]):
        super().__init__()
        self.kdtree_results = kdtree_results

    def _do(self, problem, n_samples, **kwargs):
        # pymoo yêu cầu trả về một mảng numpy 1 chiều có dtype là object
        X = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            # Nếu pop_size của NSGA-II lớn hơn số lượng KDTree tìm được (K), 
            # chúng ta sẽ lặp lại các mẫu này để điền đầy quần thể
            idx = i % len(self.kdtree_results)
            X[i] = self.kdtree_results[idx]
            
        return X

if __name__ == "__main__":
    print("--- BẮT ĐẦU TEST BƯỚC 4: KDTREE INITIALIZER ---")
    
    # 1. Tạo tập dữ liệu Dummy (Có thêm cột nhãn 'target')
    data = {
        "duration": [6, 48, 12, 42, 24, 36, 18],
        "credit_amount": [1169, 5951, 2096, 7882, 4870, 3000, 1500],
        "age": [67, 22, 49, 45, 53, 30, 40],
        "purpose": ["radio/tv", "education", "furniture", "car", "radio/tv", "car", "education"],
        "target": [1, 0, 1, 0, 1, 1, 0] # 1 = Duyệt, 0 = Từ chối
    }
    X_train = pd.DataFrame(data)
    
    # 2. Khởi tạo và Fit Preprocessor (Chỉ dùng các cột features, không dùng cột target)
    X_features = X_train.drop(columns=["target"])
    preprocessor = CreditPreprocessor(dataset_name="german", model_type="embedding")
    preprocessor.fit(X_features)
    
    # 3. Test KDTreeInitializer
    print("\nKhởi tạo KDTree với dữ liệu đã được duyệt vay (Nhãn = 1)...")
    kdtree_engine = KDTreeInitializer(
        df_train=X_train, 
        preprocessor=preprocessor, 
        target_col="target", 
        good_label=1
    )
    
    # Khách hàng bị từ chối (Lấy dòng index 1, target = 0)
    x_rejected = X_features.iloc[[1]].copy()
    print("\n[Hồ sơ bị từ chối (Gốc)]:")
    print(x_rejected.to_dict(orient='records')[0])
    
    # Lấy 2 láng giềng tốt nhất
    print("\nTìm kiếm 2 láng giềng gần nhất đã được duyệt vay...")
    kdtree_pop = kdtree_engine.get_initial_population(x_rejected, k=2)
    
    for i, record in enumerate(kdtree_pop):
        print(f"Láng giềng {i+1}: {record}")
        
    # 4. Test KDTreeMixedSampling (Giả lập pymoo gọi hàm)
    print("\nKiểm tra lớp bọc KDTreeMixedSampling cho pymoo...")
    sampling = KDTreeMixedSampling(kdtree_pop)
    
    # Giả sử pymoo muốn khởi tạo quần thể gồm 5 cá thể (pop_size = 5)
    # Vì KDTree chỉ tìm được 2 láng giềng, Sampling phải lặp lại dữ liệu để điền đủ 5
    mock_population = sampling._do(problem=None, n_samples=5)
    
    print(f"Kích thước quần thể sinh ra: {mock_population.shape}")
    print(f"Kiểu dữ liệu của mảng: {mock_population.dtype} (Phải là 'object')")
    print("Phần tử đầu tiên trong quần thể:", mock_population[0])
    print("Phần tử thứ 3 (phải lặp lại từ đầu):", mock_population[2])
    
    # Assertions
    assert isinstance(kdtree_pop, list), "Lỗi: Đầu ra KDTree không phải là list."
    assert mock_population.dtype == object, "Lỗi: Mảng Sampling không phải là dtype=object."
    assert mock_population.shape[0] == 5, "Lỗi: Sampling không sinh đủ số lượng cá thể yêu cầu."
    
    print("\n=> CHÚC MỪNG! Module KDTree và Sampling đã hoạt động hoàn hảo.")