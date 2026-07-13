# usage: 
#   python -m src.cf.single_cf_explainer --dataset german_credit --n-tests 10
#   python -m src.cf.single_cf_explainer --dataset lending_club --n-tests 10
#   python -m src.cf.single_cf_explainer --dataset gmsc --n-tests 10

import argparse
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import torch
from typing import Any  
from sklearn.neighbors import LocalOutlierFactor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embed_mlp import EmbedMLP
from src.models.model_wrapper import EmbedMLPWrapper
from src.data_processing.preprocess import CreditPreprocessor
from src.cf.metric_evaluator import CFMEvaluator
from src.data_processing.utils import load_data, split_data, get_target_col

# Import các module tối ưu của thư viện pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"


def _load_decision_threshold(dataset: str) -> float:
    eval_path = OUTPUTS_DIR / dataset / "models" / "eval_results.json"
    if not eval_path.exists():
        return 0.5

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_json = json.load(f)

    threshold = eval_json.get("embed_mlp", {}).get("best_threshold", 0.5)
    return float(threshold or 0.5)

# SINGLE-CF (WACHTER)

class SingleCFProblem(ElementwiseProblem):
    def __init__(self, x_original_scaled, wrapper, xl, xu, feature_names, feature_weights=None):
        """
        x_original_scaled: Vector hồ sơ gốc sau khi transform (numpy array)
        wrapper: Bộ bọc mô hình EmbedMLPWrapper của nhóm
        xl, xu: Giới hạn không gian tìm kiếm đặc trưng
        feature_names: Danh sách tên cột đặc trưng gốc cần thiết cho wrapper
        feature_weights: Trọng số lấy từ AcME (nếu không có, mặc định gán bằng 1)
        """
        self.x_orig = x_original_scaled
        self.wrapper = wrapper
        self.feature_names = feature_names  
        
        if feature_weights is not None:
            self.w = feature_weights
        else:
            self.w = np.ones(len(x_original_scaled))

        # n_obj=2: Chỉ giữ lại yloss và dist (Bỏ Sparsity, RIM, LOF) 
        # n_ieq_constr=0: Hoàn toàn không ép bất kỳ ràng buộc nhân quả cứng nào 
        super().__init__(n_var=len(x_original_scaled), n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        x_df = pd.DataFrame([x], columns=self.feature_names)
        
        # Nhận về xác suất prob thuộc về lớp 1 (Được duyệt vay) từ mạng nơ-ron
        y_pred = self.wrapper.predict_proba(x_df)[0]
        
        # Hàm mục tiêu 1: yloss (Hinge loss hướng tới lật nhãn sang 1) 
        f1_yloss = max(0.0, 1.0 - y_pred)
        
        # Hàm mục tiêu 2: dist (Khoảng cách tiệm cận đặc trưng cơ bản sử dụng Euclidean có trọng số) 
        f2_dist = np.sqrt(np.sum(((x - self.x_orig) ** 2) * (1.0 / self.w)))
        
        # Nạp mảng tối ưu
        out["F"] = [f1_yloss, f2_dist]


# Kết nối với run_experiments class SingleCFGenerator:
    def __init__(self, model_wrapper, df_train_raw, target_col="target"):
        """
        Khởi tạo Generator cho Single-CF (Wachter Baseline) tương thích với cấu hình thực nghiệm.
        """
        self.wrapper = model_wrapper
        self.target_col = target_col
        self.feature_names = [col for col in df_train_raw.columns if col != target_col]
        
        # Trích xuất preprocessor từ wrapper để thực hiện encode/decode dữ liệu
        self.preprocessor = model_wrapper.preprocessor
        
        # Khởi tạo không gian tìm kiếm bounds [0, 1] dựa trên số lượng đặc trưng sau biến đổi
        # Lấy mẫu 1 dòng từ df_train_raw để kiểm tra kích thước đầu ra
        X_sample = df_train_raw.drop(columns=[target_col], errors='ignore').head(1)
        dummy_transform = self.preprocessor.transform(X_sample)
        n_features_scaled = dummy_transform.shape[1]
        
        self.xl = np.zeros(n_features_scaled)
        self.xu = np.ones(n_features_scaled)

    def generate(self, x_req: pd.Series, pop_size: int = 100, n_gen: int = 50, num_cf: int = 3) -> pd.DataFrame:
        """
        Sinh Counterfactual cho một hồ sơ khách hàng đơn lẻ (x_req).
        Giao diện hàm này đồng bộ hoàn toàn với cấu trúc gọi từ run_experiments.py.
        """
        # 1. Chuyển đổi Series thành DataFrame 1 dòng và transform sang không gian scaled
        x_req_df = x_req.to_frame().T if isinstance(x_req, pd.Series) else pd.DataFrame([x_req])
        # Đảm bảo loại bỏ các cột không liên quan nếu có
        x_req_df = x_req_df.drop(columns=[self.target_col], errors='ignore').drop(columns=["y_prob"], errors='ignore')
        
        x_req_scaled = self.preprocessor.transform(x_req_df)[0]
        
        # 2. Khởi tạo SingleCFProblem
        problem = SingleCFProblem(
            x_original_scaled=x_req_scaled,
            wrapper=self.wrapper,
            xl=self.xl,
            xu=self.xu,
            feature_names=self.feature_names,
            feature_weights=None
        )
        
        # 3. Cấu hình NSGA-II 
        algorithm = NSGA2(
            pop_size=pop_size, 
            crossover=SBX(prob=0.9, eta=15), 
            mutation=PM(prob=0.1, eta=20) 
        )
        
        # 4. Thực thi tối ưu hóa tìm Pareto Front
        res = minimize(problem, algorithm, termination=('n_gen', n_gen), seed=42, verbose=False)
        
        if res.X is None or len(res.X) == 0:
            return pd.DataFrame() # Trả về DF rỗng nếu không tìm thấy nghiệm
            
        # 5. Giải nén tập nghiệm Pareto và lấy số lượng ứng viên theo yêu cầu num_cf
        best_candidates = res.X[:num_cf] if res.X.ndim > 1 else np.array([res.X])
        
        # 6. Biến đổi ngược về dạng dữ liệu thô (raw DataFrame) ban đầu
        cf_results = self.preprocessor.inverse_transform(best_candidates)
        
        return cf_results

# Luồng tải mô hình hệ thống

def _load_embed_model(dataset: str, device: torch.device) -> tuple[EmbedMLP, dict[str, Any], float]:
    cfg_path = OUTPUTS_DIR / dataset / "models" / "best_configs.json"
    model_path = OUTPUTS_DIR / dataset / "models" / "embed_mlp_best.pkl"
    eval_path = OUTPUTS_DIR / dataset / "models" / "eval_results.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy best config tại: {cfg_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model embed_mlp tại: {model_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    best_cfg = cfg_json.get("embed_mlp", {})
    if not best_cfg:
        raise ValueError(f"best_configs.json không có cấu hình cho embed_mlp của bộ {dataset}.")

    decision_threshold = 0.5
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_json = json.load(f)
        decision_threshold = float(eval_json.get("embed_mlp", {}).get("best_threshold", 0.5) or 0.5)

    model = EmbedMLP(
        input_num_dim=int(best_cfg["input_num_dim"]),
        cat_dims=list(best_cfg["cat_dims"]),
        emb_dims=best_cfg.get("emb_dims", None),
        hidden_dims=(int(best_cfg["hidden_h1"]), int(best_cfg["hidden_h2"])),
        dropout=float(best_cfg.get("dropout", 0.3)),
    )

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
        
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        raise ValueError("embed_mlp_best.pkl không phải state_dict như mong đợi.")

    model.to(device)
    model.eval()
    return model, best_cfg, decision_threshold

# MAIN WORKFLOW

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation for SingleCF (Wachter Baseline).")
    parser.add_argument("--dataset", type=str, default="german_credit", choices=["german_credit", "lending_club", "gmsc"])
    parser.add_argument("--n-tests", type=int, default=10, help="Number of rejected profiles to test.")
    args = parser.parse_args()

    dataset_name = args.dataset
    target_col = get_target_col(dataset_name)
    decision_threshold = _load_decision_threshold(dataset_name)

    print("=" * 70)
    print(f"Running baseline: single-cf (wachter) on: {dataset_name.upper()}")
    print(f"Decision threshold: {decision_threshold:.6f}")
    print("=" * 70)

    df = load_data(dataset_name)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df=df, target_col=target_col)
    df_train_raw = pd.concat([X_train, y_train], axis=1)

    preprocessor = CreditPreprocessor(dataset_name=dataset_name, model_type='embedding')
    preprocessor.fit(X_train=X_train)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_cfg, decision_threshold = _load_embed_model(dataset_name, device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)

    # SỬA LỖI: Khởi tạo mô hình LOF cục bộ phục vụ cho việc đánh giá LOF-NR 
    print("\n[SỬA LỖI] Đang huấn luyện mô hình Local Outlier Factor (LOF) cho Evaluator...")
    X_train_scaled = preprocessor.transform(X_train)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, metric='minkowski') 
    lof_model.fit(X_train_scaled)
    print("Huấn luyện xong mô hình LOF.")

    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=lof_model,  
        df_train_raw=df_train_raw,
        decision_threshold=decision_threshold,
    )

    rejected_label = 0
    valid_rejected_instances = []
    for count, (idx, row) in enumerate(X_test.iterrows()):
        actual_label = y_test.iloc[count]
        if int(actual_label) == rejected_label: 
            prob = wrapper.predict_proba(row.to_frame().T)[0]
            if prob < decision_threshold:
                valid_rejected_instances.append((row, prob))
        if len(valid_rejected_instances) == args.n_tests:
            break

    print(f"Đã tìm thấy {len(valid_rejected_instances)} khách hàng bị TỪ CHỐI thực tế để test.")

    all_metrics = []
    success_count = 0

    # Lấy không gian bounds sau khi preprocessor mã hóa để giới hạn quần thể di truyền
    dummy_transform = preprocessor.transform(X_train.head(1))
    n_features_scaled = dummy_transform.shape[1]
    xl = np.zeros(n_features_scaled)
    xu = np.ones(n_features_scaled)

    for i, (x_req, original_prob) in enumerate(valid_rejected_instances):
        print(f"\n--- Đang xử lý Hồ sơ #{i+1}/{len(valid_rejected_instances)} (Prob gốc: {original_prob:.4f}) ---")
        
        # 1. Mã hóa hồ sơ gốc sang không gian vector để thuật toán tối ưu tiến hóa
        x_req_scaled = preprocessor.transform(x_req.to_frame().T)[0]
        
        # 2. Khởi tạo bài toán tối ưu SingleCF 
        problem = SingleCFProblem(
            x_original_scaled=x_req_scaled,
            wrapper=wrapper,
            xl=xl,
            xu=xu,
            feature_names=X_train.columns,  
            feature_weights=None 
        )
        
        # 3. Cấu hình cấu trúc NSGA-II với kích thước tương tự mô hình chính để đối chứng công bằng cit
        algorithm = NSGA2(
            pop_size=100, 
            crossover=SBX(prob=0.9, eta=15), 
            mutation=PM(prob=0.1, eta=20) 
        )
        
        # 4. Tìm kiếm nghiệm Pareto (Chạy 50 thế hệ đồng bộ)
        res = minimize(problem, algorithm, termination=('n_gen', 50), seed=42, verbose=False)
        
        if res.X is None or len(res.X) == 0:
            print(" [!] Thất bại: Không tìm thấy Giải pháp Counterfactual.")
            continue
            
        print(f" [*] Thành công: Tìm thấy các ứng viên tốt từ Pareto Front.")
        success_count += 1
        
        # 5. Giải nén tập nghiệm Pareto (lấy tối đa 3 mẫu tốt nhất) và giải mã ngược về dạng DataFrame gốc
        best_candidates = res.X[:3] if res.X.ndim > 1 else np.array([res.X])
        cf_results = preprocessor.inverse_transform(best_candidates)
        
        # 6. Đánh giá chất lượng bằng bộ chỉ số chung của hệ thống 
        metrics = evaluator.evaluate(x_original=x_req, cf_df=cf_results)
        all_metrics.append(metrics)

    print("\n" + "="*50)
    print(f" Kết quả benchmark tổng quát (single-cf baseline) ")
    print("="*50)
    
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        for key, value in avg_metrics.items():
            print(f" * {key}: {value:.4f}")

if __name__ == "__main__":
    main()