import dice_ml
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import torch
from typing import Any
import json

# Thiết lập đường dẫn tương đối để import module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Import các module của bạn
from src.preprocess.preprocess import CreditPreprocessor
from src.cf.cfm_fm.core.generator import CFMFWGenerator
from src.cf.cfm_fm.core.evaluator import CFMEvaluator
from src.cf.cfm_fm.core.model_wrapper import EmbedMLPWrapper
from src.models.embed_mlp import EmbedMLP
from sklearn.model_selection import train_test_split

# ==========================================
# PHẦN 1: CÁC HÀM TẢI MÔ HÌNH
# ==========================================
def _load_embed_model(dataset: str, device: torch.device) -> tuple[EmbedMLP, dict[str, Any]]:
    cfg_path = RESULTS_DIR / dataset / "best_configs.json"
    model_path = MODELS_DIR / dataset / "embed_mlp_best.pkl"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy best config tại: {cfg_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model embed_mlp tại: {model_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_json = json.load(f)

    best_cfg = cfg_json.get("embed_mlp", {}).get("best_config", {})
    if not best_cfg:
        raise ValueError("best_configs.json không có cấu hình cho embed_mlp.")

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
    return model, best_cfg

# ==========================================
# PHẦN 2: BỘ CHUYỂN ĐỔI (ADAPTER) CHO DICE
# ==========================================
class DiCEModelAdapter:
    """
    Bộ chuyển đổi giúp EmbedMLPWrapper giao tiếp chuẩn mực với DiCE (Scikit-learn API).
    """
    def __init__(self, base_wrapper):
        self.base_wrapper = base_wrapper

    def predict_proba(self, X) -> np.ndarray:
        """Trả về ma trận (n_samples, 2) cho DiCE"""
        # Đảm bảo đầu vào luôn là DataFrame để tránh lỗi ép kiểu của NumPy/Pandas
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        probs_class_1 = []
        for i in range(len(X)):
            # QUAN TRỌNG: Chuyển dòng dữ liệu thành dạng Dictionary 
            # để đảm bảo Preprocessor đọc đúng cặp {Tên cột: Giá trị}
            row_dict = X.iloc[i].to_dict()
            
            # Đưa vào wrapper của bạn
            prob = float(self.base_wrapper.predict_proba(row_dict)[0])
            probs_class_1.append(prob)
            
        probs_class_1 = np.array(probs_class_1)
        probs_class_0 = 1.0 - probs_class_1
        
        # Ghép thành ma trận 2 cột [[P(Class=0), P(Class=1)], ...]
        return np.vstack((probs_class_0, probs_class_1)).T

    def predict(self, X) -> np.ndarray:
        """Trả về mảng nhãn 0/1 cho DiCE"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)

# ==========================================
# PHẦN 3: HÀM CHẠY DiCE
# ==========================================
def run_dice_benchmark(model_wrapper, df_train, x_rejected_df, target_col='Class'):
    # Lấy danh sách các biến ĐƯỢC PHÉP thay đổi từ Preprocessor của bạn
    actionable_features = model_wrapper.preprocessor.configs["german"]["actionable"]
    
    num_features = model_wrapper.preprocessor.get_metadata()["num_features"]
    d = dice_ml.Data(dataframe=df_train, continuous_features=num_features, outcome_name=target_col)
    
    dice_adapter = DiCEModelAdapter(model_wrapper)
    m = dice_ml.Model(model=dice_adapter, backend="sklearn")
    
    exp = dice_ml.Dice(d, m, method="genetic")
    
    dice_results = []
    for i in range(len(x_rejected_df)):
        query_instance = x_rejected_df.iloc[[i]].drop(columns=[target_col], errors='ignore')
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance, 
                total_CFs=3, 
                desired_class="opposite",
                # QUAN TRỌNG: Chỉ cho phép DiCE thay đổi những gì bạn cho phép
                features_to_vary=actionable_features, 
                verbose=False
            )
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            cf_df = cf_df.drop(columns=[target_col], errors='ignore')
            dice_results.append(cf_df)
        except Exception as e:
            print(f"Mẫu {i+1} - Lỗi DiCE: {str(e)}")
            dice_results.append(None)
            
    return dice_results

# ==========================================
# PHẦN 4: HÀM ĐIỀU PHỐI BENCHMARK
# ==========================================
def execute_benchmarking(generator_cfm, wrapper, df_train, df_test, n_samples=20):
    print("\n" + "="*70)
    print(f" BẮT ĐẦU BENCHMARK: CFM-FW (Ours) vs DiCE (Microsoft)")
    print("="*70)
    
    test_rejected = df_test[df_test['Class'] == 0].head(n_samples)
    
    # Lấy trọng số AcME một lần duy nhất
    weights = generator_cfm.acme.explain(df_train.drop(columns=['Class']))["abs_weights"]
    evaluator = CFMEvaluator(
        preprocessor=wrapper.preprocessor, 
        df_train=df_train, 
        feature_weights=weights,
        target_col='Class'
    )

    all_stats = []

    for i in range(len(test_rejected)):
        x_orig = test_rejected.iloc[i].drop('Class')
        print(f"\n--- Đang xử lý mẫu {i+1}/{n_samples} ---")
        
        # 1. CFM-FW (Ours)
        start_cfm = time.time()
        cf_cfm = generator_cfm.generate(x_orig, pop_size=100, n_gen=50)
        time_cfm = time.time() - start_cfm
        
        if cf_cfm is not None and not cf_cfm.empty:
            m_cfm = evaluator.evaluate(x_orig, cf_cfm)
            m_cfm['Method'] = 'CFM-FW (Ours)'
            m_cfm['Runtime (s)'] = time_cfm
            all_stats.append(m_cfm)
        
        # 2. DiCE (Baseline)
        start_dice = time.time()
        dice_list = run_dice_benchmark(wrapper, df_train, test_rejected.iloc[[i]], target_col='Class')
        cf_dice = dice_list[0]
        time_dice = time.time() - start_dice
        
        if cf_dice is not None and not cf_dice.empty:
            m_dice = evaluator.evaluate(x_orig, cf_dice)
            m_dice['Method'] = 'DiCE (Baseline)'
            m_dice['Runtime (s)'] = time_dice
            all_stats.append(m_dice)

    # TỔNG HỢP
    if not all_stats:
        print("\nKhông có phương pháp nào sinh được giải pháp để so sánh!")
        return pd.DataFrame()

    df_stats = pd.DataFrame(all_stats)
    
    # Lọc các cột số để tính trung bình
    numeric_cols = df_stats.select_dtypes(include=[np.number]).columns.tolist()
    
    final_report = df_stats.groupby('Method')[numeric_cols].mean().reset_index()

    print("\n" + "="*70)
    print(" BÁO CÁO SO SÁNH CUỐI CÙNG (TRUNG BÌNH)")
    print("="*70)
    print(final_report.to_string(index=False))
    
    return final_report

# ==========================================
# PHẦN 5: CHẠY THỰC THI (MAIN)
# ==========================================
if __name__ == "__main__":
    print("\n[1/5] Load dữ liệu...")
    dataset = pd.read_csv(PROJECT_ROOT / "data/german_credit.csv")
    print(f"✓ Loaded {len(dataset)} records")
    print(f"  Class distribution: {dataset['Class'].value_counts().to_dict()}")

    y = dataset["Class"].values
    X = dataset.drop(columns=["Class"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    df_train_full = X_train.copy()
    df_train_full["Class"] = y_train
    df_test_full = X_test.copy()
    df_test_full["Class"] = y_test

    print("\n[2/5] Khởi tạo Preprocessor & model...")
    preprocessor = CreditPreprocessor(dataset_name="german", model_type="embedding")
    preprocessor.fit(X_train)
    metadata = preprocessor.get_metadata()
    print(f"✓ Num features: {len(metadata['num_features'])}, Cat features: {len(metadata['cat_features'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, best_cfg = _load_embed_model("germancredit", device)
        wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)
        print(f"✓ Model loaded on device: {device}")
    except FileNotFoundError as e:
        print(f"✗ Model không tìm thấy: {e}")
        print("  Thoát chương trình. Vui lòng kiểm tra lại đường dẫn model.")
        sys.exit(1)

    print("\n[3/5] Khởi tạo CFM-FW Generator...")
    cfm_generator = CFMFWGenerator(
        model_wrapper=wrapper,
        df_train=df_train_full,
        preprocessor=preprocessor
    )
    print("✓ CFM-FW Generator initialized successfully.")

    print("\n[4/5] Thực thi benchmark giữa CFM-FW và DiCE...")
    # Chạy thử trên 20 mẫu (bạn có thể đổi thành 50 nếu muốn số liệu dày hơn)
    report = execute_benchmarking(cfm_generator, wrapper, df_train_full, df_test_full, n_samples=20)

    if not report.empty:
        print("\n[5/5] Đang lưu báo cáo...")
        save_path = RESULTS_DIR / "final_comparison.csv"
        # Đảm bảo thư mục tồn tại trước khi lưu
        save_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(save_path, index=False)
        print(f"✓ Benchmarking completed. Report saved to {save_path}")