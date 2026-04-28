# usage: python -m src.cf.dice_explainer

import pandas as pd
import numpy as np
from dice_ml import Data, Model, Dice
import torch
import sys
from pathlib import Path
from typing import Any
import json

from src.models.embed_mlp import EmbedMLP
from src.models.model_wrapper import EmbedMLPWrapper
from src.data_processing.preprocess import CreditPreprocessor
from src.cf.metric_evaluator import CFMEvaluator
from src.cf.cfm_fm.lof import PlausibilityLOF # Import class LOF của bạn
from src.data_processing.utils import load_data, split_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    print("[1] Khởi tạo DiCE Data...")
    d = Data(
        dataframe=df_train,
        continuous_features=num_features,
        outcome_name=target_col
    )

    # 2. KHỞI TẠO DiCE MODEL (Adapter)
    print("[2] Khởi tạo DiCE Model (Bọc Black-box Wrapper)...")
    class DiCEAdapter:
        def __init__(self, wrapper_instance, feature_names):
            self.wrapper = wrapper_instance
            self.feature_names = feature_names
            
        def predict_proba(self, X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)
            probs = self.wrapper.predict_proba(X)
            probs = np.asarray(probs)
            if probs.ndim == 1:
                probs_2d = np.zeros((len(probs), 2))
                probs_2d[:, 1] = probs
                probs_2d[:, 0] = 1.0 - probs
                return probs_2d
            return probs
            
        def predict(self, X):
            probs = self.predict_proba(X)
            return (probs[:, 1] >= 0.5).astype(int)

    m = Model(model=DiCEAdapter(wrapper, d.feature_names), backend="sklearn")

    # 3. KHỞI TẠO DiCE EXPLAINER
    print("[3] Khởi tạo DiCE Explainer (Genetic Algorithm)...")
    exp = Dice(d, m, method="genetic")

    # 4. KHỞI TẠO CFM EVALUATOR (Dùng chung thước đo)
    print("[4] Khởi tạo CFM Evaluator...")
    X_train_raw = df_train.drop(columns=[target_col])
    X_train_scaled = preprocessor.transform(X_train_raw)
    
    plausibility_module = PlausibilityLOF(n_neighbors=20)
    plausibility_module.fit(X_train_scaled)

    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=plausibility_module,
        df_train_raw=X_train_raw
    )

    # 5. SINH CF & ĐÁNH GIÁ
    print("\n[5] Bắt đầu sinh CF bằng DiCE và Đánh giá...")
    potential_rejects = df_test[df_test[target_col] == 0]
    
    all_metrics = []
    success_count = 0
    N_TESTS = 10 

    for i, (idx, row) in enumerate(potential_rejects.iterrows()):
        if len(all_metrics) >= N_TESTS:
            break
            
        # Tách features ra khỏi target
        x_req_with_target = row.to_frame().T
        x_req = x_req_with_target.drop(columns=[target_col])
        
        prob = wrapper.predict_proba(x_req)[0]
        if prob >= 0.5: continue
            
        print(f"--- Hồ sơ #{len(all_metrics)+1} (Prob: {prob:.4f}) ---")
        
        try:
            dice_exp = exp.generate_counterfactuals(
                x_req, 
                total_CFs=num_cf, 
                desired_class="opposite",
                features_to_vary=actionable_features
            )
            
            dice_raw_df = dice_exp.cf_examples_list[0].final_cfs_df
            if dice_raw_df is not None and not dice_raw_df.empty:
                success_count += 1
                cf_df_clean = dice_raw_df.drop(columns=[target_col])
                cf_df_clean['predicted_prob'] = wrapper.predict_proba(cf_df_clean)

                metrics = evaluator.evaluate(x_original=row.drop(target_col), cf_df=cf_df_clean)
                all_metrics.append(metrics)
            else:
                print(" [!] DiCE không tìm thấy CF.")
        except Exception as e:
            print(f" [!] Lỗi: {e}")

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

    # [STEP 1] Load data
    print("\n[STEP 1] Loading German Credit data...")
    dataset_name = 'german_credit'
    df = load_data(dataset_name)
    target_col = 'Class'
    print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")

    # [STEP 2] Splitting data - Cập nhật để nhận đủ 6 biến từ utils.py
    print("\n[STEP 2] Splitting data (80/20 train/test)...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df, 
        target_col=target_col, 
        dataset_name=dataset_name, 
        random_state=42
    )
    
    # Ghép lại thành df_train/test chứa target vì thư viện DiCE yêu cầu cột nhãn trong dataframe
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    print(f"✅ Train features: {X_train.shape}, Test features: {X_test.shape}")

    # [STEP 3] Preprocessing
    print("\n[STEP 3] Preprocessing with CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name=dataset_name, model_type='embedding')
    preprocessor.fit(X_train=X_train) 
    
    # FIX LỖI: Gán giá trị cho biến metadata
    metadata = preprocessor.get_metadata() 
    print(f"✅ Num features: {len(metadata['num_features'])}")
    print(f"✅ Cat features: {len(metadata['cat_features'])}")

    # [STEP 4] Load trained model
    print("\n[STEP 4] Load trained EmbedMLP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_cfg = _load_embed_model(dataset_name, device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)
    print(f"✅ Loaded EmbedMLP model.")

    # [STEP 5] Run Benchmark
    run_dice_benchmark(
        df_train=df_train, 
        df_test=df_test, 
        wrapper=wrapper, 
        preprocessor=preprocessor, 
        target_col=target_col, 
        num_cf=3
    )