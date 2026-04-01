import pandas as pd
import numpy as np
from dice_ml import Data, Model, Dice
import torch
import sys
from pathlib import Path
from typing import Any
import json
# Allow running this file directly: `python testwraper.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embed_mlp import EmbedMLP
from src.models.model_wrapper import EmbedMLPWrapper
from src.preprocess.preprocess import CreditPreprocessor
from src.cf.metric_evaluator import CFMEvaluator
from src.cf.cfm_fm.lof import PlausibilityLOF # Import class LOF của bạn
from src.preprocess.utils import load_data, split_data

OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Đảm bảo import các class hiện tại của bạn
# from src.cf.cfm_fm.preprocessor import CreditPreprocessor
# from src.cf.cfm_fm.evaluator import CFMEvaluator
# from src.cf.cfm_fm.core.model_wrapper import EmbedMLPWrapper
# from src.cf.cfm_fm.generator import PlausibilityLOF # Nếu Evaluator cần

def run_dice_benchmark(df_train: pd.DataFrame, df_test: pd.DataFrame, wrapper, preprocessor, target_col="Class", num_cf=3):
    print("\n" + "="*50)
    print(" CHẠY BENCHMARK VỚI DiCE (BASELINE) ")
    print("="*50)

    metadata = preprocessor.get_metadata()
    num_features = metadata["num_features"]
    actionable_features = metadata["actionable"]

    # 1. KHỞI TẠO DiCE DATA
    # DiCE yêu cầu Dataframe phải chứa cả cột Target
    print("[1] Khởi tạo DiCE Data...")
    d = Data(
        dataframe=df_train,
        continuous_features=num_features,
        outcome_name=target_col
    )

    # 2. KHỞI TẠO DiCE MODEL
    # Sử dụng backend "sklearn" vì wrapper của chúng ta đã có sẵn hàm predict_proba
    print("[2] Khởi tạo DiCE Model (Bọc Black-box Wrapper)...")
    
    # Tạo một class Adapter nhỏ để DiCE hiểu wrapper của bạn
    class DiCEAdapter:
        def __init__(self, wrapper_instance, feature_names):
            self.wrapper = wrapper_instance
            self.feature_names = feature_names
            
        def predict_proba(self, X):
            # 1. Đảm bảo đầu vào là DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)
                
            # 2. Lấy dự đoán từ mô hình của bạn
            probs = self.wrapper.predict_proba(X)
            probs = np.asarray(probs)
            
            # 3. ÉP KIỂU VỀ 2D ARRAY CHO DiCE
            if probs.ndim == 1:
                # Nếu là mảng 1D (chỉ có xác suất nhãn 1)
                probs_2d = np.zeros((len(probs), 2))
                probs_2d[:, 1] = probs          # Cột 1: Xác suất duyệt (Class 1)
                probs_2d[:, 0] = 1.0 - probs    # Cột 0: Xác suất từ chối (Class 0)
                return probs_2d
            
            return probs
            
        def predict(self, X):
            probs = self.predict_proba(X)
            # Bây giờ probs chắc chắn là 2D, việc gọi [:, 1] là hoàn toàn an toàn
            return (probs[:, 1] >= 0.5).astype(int)

    m = Model(model=DiCEAdapter(wrapper, d.feature_names), backend="sklearn")

    # 3. KHỞI TẠO DiCE EXPLAINER
    # Dùng thuật toán Genetic (Tiến hóa) để công bằng khi so sánh với NSGA-II của CFM-FW
    print("[3] Khởi tạo DiCE Explainer (Genetic Algorithm)...")
    exp = Dice(d, m, method="genetic")

    # 4. KHỞI TẠO CFM EVALUATOR (Dùng chung thước đo)
    print("[4] Khởi tạo CFM Evaluator...")
    # Bắt buộc phải drop target để đưa X_train_raw vào huấn luyện LOF
    X_train_raw = df_train.drop(columns=[target_col])
    
    # Khởi tạo nhanh LOF để chấm điểm Plausibility cho DiCE
    X_train_scaled = preprocessor.transform(X_train_raw)
    plausibility_module = PlausibilityLOF(n_neighbors=20)
    plausibility_module.fit(X_train_scaled)

    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=plausibility_module,
        df_train_raw=X_train_raw
    )

    # 5. TÌM KHÁCH HÀNG BỊ TỪ CHỐI & SINH CF
    print("\n[5] Bắt đầu sinh CF bằng DiCE và Đánh giá...")
    potential_rejects = df_test[df_test[target_col] == 0].drop(columns=[target_col])
    
    all_metrics = []
    success_count = 0
    N_TESTS = 10 # Số lượng hồ sơ muốn test

    for i, (idx, row) in enumerate(potential_rejects.iterrows()):
        if len(all_metrics) >= N_TESTS:
            break
            
        x_req = row.to_frame().T
        prob = wrapper.predict_proba(x_req)[0, 1] if wrapper.predict_proba(x_req).ndim == 2 else wrapper.predict_proba(x_req)[0]
        
        if prob >= 0.5:
            continue # Chỉ test người thực sự bị từ chối
            
        print(f"\n--- Hồ sơ #{len(all_metrics)+1} (Prob: {prob:.4f}) ---")
        
        try:
            # Sinh CF: Áp dụng features_to_vary để Khóa các biến Immutable
            dice_exp = exp.generate_counterfactuals(
                x_req, 
                total_CFs=num_cf, 
                desired_class="opposite",
                features_to_vary=actionable_features # <--- ÉP RÀNG BUỘC TẠI ĐÂY
            )
            
            # Trích xuất DataFrame kết quả từ DiCE
            dice_raw_df = dice_exp.cf_examples_list[0].final_cfs_df
            
            if dice_raw_df is not None and not dice_raw_df.empty:
                print(f" [*] DiCE tìm thấy {len(dice_raw_df)} CFs.")
                success_count += 1
                
                # Bỏ cột target do DiCE tự động thêm vào để tương thích với Evaluator
                cf_df_clean = dice_raw_df.drop(columns=[target_col])
                
                # Có thể thêm predicted_prob vào để Evaluator chấm Validity chính xác
                cf_df_clean['predicted_prob'] = wrapper.predict_proba(cf_df_clean)[:, 1] if wrapper.predict_proba(cf_df_clean).ndim == 2 else wrapper.predict_proba(cf_df_clean)

                # Chấm điểm bằng Evaluator của bạn!
                metrics = evaluator.evaluate(x_original=row, cf_df=cf_df_clean)
                all_metrics.append(metrics)
            else:
                print(" [!] DiCE thất bại: Không tìm thấy CF.")
                
        except Exception as e:
            print(f" [!] Lỗi khi chạy DiCE: {e}")

    # 6. IN KẾT QUẢ BENCHMARK TỔNG QUÁT
    print("\n" + "="*50)
    print(" KẾT QUẢ BENCHMARK DiCE (BASELINE) ")
    print("="*50)
    print(f"Tỷ lệ thành công (Success Rate): {success_count}/{N_TESTS} ({(success_count/N_TESTS)*100:.2f}%)")

    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
            
        for key, value in avg_metrics.items():
            if key == "Total CFs Found":
                print(f" * {key}: {value:.1f}")
            else:
                print(f" * {key}: {value:.4f}")

# CÁCH GỌI HÀM (Thêm vào cuối file test của bạn):
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

if __name__ == "__main__":
    print("=" * 70)
    print("Testing CFM-FM Counterfactual Generation")
    print("=" * 70)

    print("\n[STEP 1] Loading German Credit data...")
    df = load_data('german')
    print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")

    print("\n[STEP 2] Splitting data (80/20 train/test)...")
    split_result = split_data(df, 'german', 'Class', random_state=42)
    df_train, df_test = split_result[0], split_result[1]

    X_train = df_train.drop('Class', axis=1)
    y_train = df_train['Class']
    X_test = df_test.drop('Class', axis=1)
    y_test = df_test['Class']

    print("\n[STEP 3] Preprocessing with CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name='german', model_type='embedding')
    preprocessor.fit(X_train=X_train)
    metadata = preprocessor.get_metadata()
    print(f"✅ Num features: {len(metadata['num_features'])}")
    print(f"✅ Cat features: {len(metadata['cat_features'])}")

    print("\n[STEP 4] Load trained EmbedMLP model...")
    # load file models .pklfrom MODELS_DIR/germancredit/embed_mlp_best.pkl hoặc lấy best config trong results/germancredit/best_configs.json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_cfg = _load_embed_model('germancredit', device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)

    run_dice_benchmark(df_train, df_test, wrapper, preprocessor, target_col='Class', num_cf=3)