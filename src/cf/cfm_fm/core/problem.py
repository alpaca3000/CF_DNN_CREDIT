from typing import Dict, Any, List
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
import numpy as np

def create_pymoo_space(df_train: pd.DataFrame, metadata: Dict[str, Any], x_original: pd.Series) -> Dict[str, Any]:
    space = {}
    num_features = metadata["num_features"]
    cat_features = metadata["cat_features"]
    immutable = metadata.get("immutable", [])
    
    for col in num_features:
        # Nếu là biến KHÔNG THỂ THAY ĐỔI (ví dụ: Age)
        if col in immutable:
            val = int(x_original[col]) if pd.api.types.is_integer_dtype(df_train[col]) else float(x_original[col])
            # Khóa cứng (bounds trùng nhau) -> NSGA-II sẽ không bao giờ đột biến biến này
            space[col] = Integer(bounds=(val, val)) if isinstance(val, int) else Real(bounds=(val, val))
            
        # Nếu là biến ĐƯỢC PHÉP THAY ĐỔI (ví dụ: Income, Loan Amount)
        else:
            min_val = float(df_train[col].min())
            max_val = float(df_train[col].max())
            
            if pd.api.types.is_integer_dtype(df_train[col]) or float(min_val).is_integer():
                space[col] = Integer(bounds=(int(min_val), int(max_val)))
            else:
                space[col] = Real(bounds=(min_val, max_val))
            
    for col in cat_features:
        if col in immutable:
            # Khóa cứng biến phân loại (ví dụ: Giới tính, Tình trạng hôn nhân)
            space[col] = Choice(options=[x_original[col]])
        else:
            unique_vals = df_train[col].dropna().unique().tolist()
            space[col] = Choice(options=unique_vals)
        
    return space

class CFMProblem(ElementwiseProblem):
    """
    Bài toán tối ưu hóa đa mục tiêu sinh Counterfactual theo khung CFM-FW.
    """
    def __init__(
        self,
        model_wrapper: Any,
        original_instance: pd.Series,
        feature_weights: pd.Series,
        metadata: Dict[str, Any],
        train_ranges: Dict[str, float],
        target_prob: float = 0.51,
        **kwargs
    ):
        self.model_wrapper = model_wrapper
        self.original = original_instance.to_dict()
        self.weights = feature_weights.to_dict()
        self.target_prob = target_prob
        
        self.num_features = metadata["num_features"]
        self.cat_features = metadata["cat_features"]
        self.immutable = metadata.get("immutable", [])
        self.train_ranges = train_ranges

        # Khởi tạo bài toán với 3 mục tiêu (F) và số lượng ràng buộc (G) tương ứng với biến immutable
        super().__init__(n_obj=3, n_ieq_constr=len(self.immutable), **kwargs)

    def _evaluate(self, x: Dict[str, Any], out: Dict[str, Any], *args, **kwargs):
        """
        Đánh giá một hồ sơ phản thực tế ứng viên `x`.
        """
        # ==========================================
        # MỤC TIÊU 1 (f1): Tính hợp lệ (Validity)
        # ==========================================
        # ModelWrapper sẽ tự động bọc dict x lại và xử lý
        prob = self.model_wrapper.predict_proba(x)[0]
        # Hàm loss: Khoảng cách đến ngưỡng (Càng nhỏ càng tốt, = 0 là đạt)
        f1_validity = max(0.0, self.target_prob - prob)

        # ==========================================
        # MỤC TIÊU 2 (f2): Chi phí thay đổi (Weighted Cost)
        # ==========================================
        f2_cost = 0.0
        
        # Khoảng cách cho biến liên tục (Weighted L1)
        for col in self.num_features:
            if x[col] != self.original[col]:
                dist = abs(x[col] - self.original[col]) / (self.train_ranges[col] + 1e-9)
                f2_cost += self.weights.get(col, 1.0) * dist
                
        # Khoảng cách cho biến phân loại (Weighted Hamming)
        for col in self.cat_features:
            if x[col] != self.original[col]:
                f2_cost += self.weights.get(col, 1.0) * 1.0

        # ==========================================
        # MỤC TIÊU 3 (f3): Độ thưa (Sparsity)
        # ==========================================
        # Đếm số lượng đặc trưng bị thay đổi
        f3_sparsity = sum(1 for col in (self.num_features + self.cat_features) if x[col] != self.original[col])

        # ==========================================
        # RÀNG BUỘC NHÂN QUẢ (g): Causal Constraints
        # Quy ước của pymoo: g <= 0 là HỢP LỆ. g > 0 là BỊ PHẠT.
        # ==========================================
        G = []
        for col in self.immutable:
            if col in self.num_features:
                # Quy tắc nghiệp vụ: Tuổi không được giảm (x['age'] >= original['age'])
                # Suy ra: original['age'] - x['age'] <= 0
                penalty = self.original[col] - x[col]
                G.append(penalty)
            elif col in self.cat_features:
                # Biến phân loại không được phép đổi (ví dụ: Tình trạng hôn nhân ban đầu)
                penalty = 1.0 if x[col] != self.original[col] else -1.0
                G.append(penalty)

        # Gán kết quả đầu ra
        out["F"] = [f1_validity, f2_cost, f3_sparsity]
        
        if len(G) > 0:
            out["G"] = G


# class MockModelWrapper:
#     def predict_proba(self, x: dict) -> np.ndarray:
#         # Giả lập: Nếu người này tăng thu nhập lên trên 3000, cho xác suất duyệt = 0.8
#         # Nếu không, xác suất = 0.3
#         if x.get("income", 0) > 3000:
#             return np.array([0.8])
#         return np.array([0.3])

# if __name__ == "__main__":
#     print("--- BẮT ĐẦU TEST BƯỚC 3: CFM PROBLEM ---")

#     # 2. Dữ liệu giả lập
#     df_train = pd.DataFrame({
#         "income": [1000, 2000, 5000, 8000],
#         "age": [20, 30, 40, 50],
#         "housing": ["rent", "own", "free", "rent"]
#     })
    
#     metadata = {
#         "num_features": ["income", "age"],
#         "cat_features": ["housing"],
#         "immutable": ["age"] # Ràng buộc: Tuổi không được giảm
#     }
    
#     train_ranges = {
#         "income": 7000.0, # 8000 - 1000
#         "age": 30.0       # 50 - 20
#     }
    
#     # Giả lập trọng số AcME (income quan trọng nhất)
#     weights = pd.Series({"income": 0.7, "age": 0.1, "housing": 0.2})
    
#     # Hồ sơ khách hàng bị từ chối
#     x_original = pd.Series({"income": 2000, "age": 25, "housing": "rent"})

#     # 3. Khởi tạo bài toán
#     print("Tạo không gian pymoo...")
#     pymoo_space = create_pymoo_space(df_train, metadata)
    
#     problem = CFMProblem(
#         model_wrapper=MockModelWrapper(),
#         original_instance=x_original,
#         feature_weights=weights,
#         metadata=metadata,
#         train_ranges=train_ranges,
#         vars=pymoo_space # Bắt buộc phải có để báo cho pymoo đây là Mixed Variables
#     )
    
#     print("\nKhởi tạo bài toán thành công! Bắt đầu đánh giá các cá thể...")

#     # 4. Test Đánh giá Cá thể
    
#     # Cá thể 1: Thay đổi HỢP LỆ (Tăng thu nhập lên 4000, tuổi giữ nguyên 25)
#     cf_valid = {"income": 4000, "age": 25, "housing": "rent"}
#     out_valid = {}
#     problem._evaluate(cf_valid, out_valid)
    
#     # Cá thể 2: Thay đổi VI PHẠM (Thu nhập giữ 2000, Tuổi giảm xuống 20)
#     cf_invalid = {"income": 2000, "age": 20, "housing": "rent"}
#     out_invalid = {}
#     problem._evaluate(cf_invalid, out_invalid)
    
#     print("\n[Cá thể 1 - Đạt chuẩn]")
#     print(f"Hồ sơ: {cf_valid}")
#     print(f"Mục tiêu (F): {out_valid['F']} -> f1=0 (Đã duyệt), f2>0 (Có chi phí đổi thu nhập), f3=1 (Đổi 1 cột)")
#     print(f"Ràng buộc (G): {out_valid['G']} -> Phải <= 0 (Không vi phạm)")
#     assert out_valid["F"][0] == 0.0, "Lỗi tính f1"
#     assert out_valid["G"][0] <= 0, "Lỗi tính ràng buộc"

#     print("\n[Cá thể 2 - Vi phạm nhân quả]")
#     print(f"Hồ sơ: {cf_invalid}")
#     print(f"Mục tiêu (F): {out_invalid['F']} -> f1>0 (Vẫn bị từ chối)")
#     print(f"Ràng buộc (G): {out_invalid['G']} -> Có giá trị > 0 (Tuổi bị giảm)")
#     assert out_invalid["G"][0] > 0, "Lỗi không bắt được vi phạm tuổi"

#     print("\n=> CHÚC MỪNG! Module Định nghĩa Bài toán (CFMProblem) hoạt động chuẩn xác.")