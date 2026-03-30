from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import numpy as np

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cf.cfm_fm.core.model_wrapper import EmbedMLPWrapper
from src.cf.cfm_fm.core.acme_weight import AcMEEngine
from src.cf.cfm_fm.core.kdtree_population import KDTreeInitializer, KDTreeMixedSampling
from src.cf.cfm_fm.core.problem import CFMProblem, create_pymoo_space
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.optimize import minimize

class CFMFWGenerator:
    """
    Bộ điều phối chính thức của khung CFM-FW.
    Thực thi toàn bộ pipeline từ khi nhận hồ sơ bị từ chối đến khi xuất ra các giải pháp phản thực tế.
    """
    def __init__(self, model_wrapper: EmbedMLPWrapper, df_train: pd.DataFrame, preprocessor: Any):
        self.wrapper = model_wrapper
        self.df_train = df_train
        self.preprocessor = preprocessor
        self.metadata = preprocessor.get_metadata()
        
        # 1. Chuẩn bị KDTree (Chỉ lấy nhóm được duyệt vay, giả sử nhãn là 'target')
        # Lưu ý: Cập nhật 'target' thành tên cột nhãn thực tế của bạn
        self.kdtree = KDTreeInitializer(
            df_train=self.df_train, 
            preprocessor=self.preprocessor, 
            target_col='Class', 
            good_label=1
        )
        
        # 2. Khởi tạo AcME Engine để sẵn sàng trích xuất trọng số
        self.acme = AcMEEngine.from_dataframe_baseline(
            predict_proba=self.wrapper.predict_proba,
            df=self.df_train.drop(columns=['Class']), # Không đưa target col vào
            num_features=self.metadata["num_features"],
            cat_features=self.metadata["cat_features"]
        )

    def generate(
        self, 
        x_rejected: pd.Series, 
        pop_size: int = 100, 
        n_gen: int = 100, 
        k_neighbors: int = 50
    ) -> pd.DataFrame:
        """
        Hàm sinh Counterfactual cho một khách hàng cụ thể.
        """
        # ==========================================
        # PHA 1: Trích xuất trọng số bằng AcME
        # ==========================================
        print("1. Đang chạy AcME để tính trọng số đặc trưng...")
        explanation = self.acme.explain(self.df_train.drop(columns=['Class']))
        feature_weights = explanation["abs_weights"]

        # ==========================================
        # PHA 2: Khởi tạo hạt giống bằng KDTree
        # ==========================================
        print("2. Đang quét KDTree tìm hồ sơ láng giềng tốt...")
        kdtree_pop = self.kdtree.get_initial_population(x_rejected.to_frame().T, k=k_neighbors)
        custom_sampling = KDTreeMixedSampling(kdtree_pop)

        # ==========================================
        # PHA 3: Thiết lập Bài toán (CFMProblem)
        # ==========================================
        print("3. Khởi tạo không gian tìm kiếm và ràng buộc nhân quả...")
        pymoo_space = create_pymoo_space(
            df_train=self.df_train, 
            metadata=self.metadata, 
            x_original=x_rejected # Khóa cứng các biến immutable bằng chính giá trị gốc
        )
        
        train_ranges = {col: float(self.df_train[col].max() - self.df_train[col].min()) 
                        for col in self.metadata["num_features"]}

        problem = CFMProblem(
            model_wrapper=self.wrapper,
            original_instance=x_rejected,
            feature_weights=feature_weights,
            metadata=self.metadata,
            train_ranges=train_ranges,
            vars=pymoo_space
        )

        # ==========================================
        # PHA 4: Khởi tạo NSGA-II (Áp dụng SBX, PM, UX theo đúng paper)
        # ==========================================
        print("4. Chạy thuật toán tiến hóa NSGA-II...")
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=custom_sampling, # Bơm KDTree vào
            # MixedVariableMating tự động dùng SBX cho biến số và UX/Point cho biến phân loại
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )

        # ==========================================
        # PHA 5: Tối ưu hóa và Trích xuất Kết quả (Pareto Front)
        # ==========================================
        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', n_gen),
            seed=42,
            verbose=False # Đổi thành True nếu bạn muốn xem log từng thế hệ
        )

        if res.X is None:
            print("Cảnh báo: Không tìm thấy hồ sơ nào thỏa mãn toàn bộ ràng buộc!")
            return pd.DataFrame()

        # res.X có thể là 1 dictionary (nếu chỉ có 1 nghiệm) hoặc mảng các dictionary
        solutions = [res.X] if isinstance(res.X, dict) else res.X.tolist()
        
        # Chuyển đổi kết quả thành DataFrame
        df_cf = pd.DataFrame(solutions)
        
        # Tính toán lại xác suất duyệt vay cho các giải pháp này để hiển thị
        probs = [self.wrapper.predict_proba(row)[0] for row in solutions]
        df_cf['predicted_prob'] = probs
        
        # Chỉ giữ lại những hồ sơ thực sự vượt qua ngưỡng duyệt (vd: > 0.5)
        df_cf = df_cf[df_cf['predicted_prob'] > 0.5].reset_index(drop=True)
        
        print(f"Hoàn thành! Tìm thấy {len(df_cf)} giải pháp phản thực tế khả thi.")
        return df_cf