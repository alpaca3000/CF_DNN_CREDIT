# usage: 
#   python -m src.cf.cfm_explainer --dataset german_credit --n-tests 10
#   python -m src.cf.cfm_explainer --dataset lending_club --n-tests 10
#   python -m src.cf.cfm_explainer --dataset gmsc --n-tests 10

import argparse
from pathlib import Path
import sys
import torch
from typing import Any
import json
import pandas as pd

# Đảm bảo PROJECT_ROOT nằm trong sys.path để tránh lỗi ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embed_mlp import EmbedMLP
from src.models.model_wrapper import EmbedMLPWrapper
from src.data_processing.preprocess import CreditPreprocessor
from src.cf.metric_evaluator import CFMEvaluator
from src.cf.cfm_fm.generator import CFMFWGenerator
from src.data_processing.utils import load_data, split_data, get_target_col

OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"

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
        raise ValueError(f"best_configs.json không có cấu hình cho embed_mlp của bộ {dataset}.")

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

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation for CFM-FW Explainer.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="german_credit", 
        choices=["german_credit", "lending_club", "gmsc"], 
        help="Dataset name to evaluate."
    )
    parser.add_argument(
        "--n-tests", 
        type=int, 
        default=10, 
        help="Number of rejected profiles to test."
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    target_col = get_target_col(dataset_name)

    print("=" * 70)
    print(f"Testing CFM-FM Counterfactual Generation on: {dataset_name.upper()}")
    print("=" * 70)

    print(f"\n[STEP 1] Loading {dataset_name} data...")
    df = load_data(dataset_name)
    print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")

    print("\n[STEP 2] Splitting data...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df=df, 
        target_col=target_col
    )
    
    # Kết hợp lại để phục vụ KDTree khởi tạo quần thể phản thực
    df_train_raw = pd.concat([X_train, y_train], axis=1)
    print(f"✅ Train features: {X_train.shape}, Test features: {X_test.shape}")

    print("\n[STEP 3] Preprocessing with CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name=dataset_name, model_type='embedding')
    preprocessor.fit(X_train=X_train)
    metadata = preprocessor.get_metadata()
    print(f"✅ Num features: {len(metadata['num_features'])}")
    print(f"✅ Cat features: {len(metadata['cat_features'])}")

    print("\n[STEP 4] Load trained EmbedMLP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_cfg = _load_embed_model(dataset_name, device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)
    print(f"✅ Loaded EmbedMLP with config: {best_cfg}")

    print("\n[STEP 5] Initialize CFM-FM Generator...")
    generator = CFMFWGenerator(
        model_wrapper=wrapper,
        df_train_raw=df_train_raw,
        target_col=target_col
    )
    print("✅ CFM-FM Generator initialized.")

    print("\n[STEP 6] & [STEP 7] BATCH EVALUATION...")
    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=generator.plausibility_module,
        df_train_raw=df_train_raw
    )

    # Do tất cả các bộ dữ liệu đã đồng bộ: 0 là Từ chối (Rejected)
    # Xác suất prob < 0.5 nghĩa là mô hình dự đoán Từ chối
    rejected_label = 0

    valid_rejected_instances = []
    # Sử dụng .iloc[count] để truy xuất chính xác theo dòng sau khi reset_index
    for count, (idx, row) in enumerate(X_test.iterrows()):
        actual_label = y_test.iloc[count]
        if int(actual_label) == rejected_label: 
            prob = wrapper.predict_proba(row.to_frame().T)[0]
            if prob < 0.5:
                valid_rejected_instances.append((row, prob))
        
        if len(valid_rejected_instances) == args.n_tests:
            break

    print(f"✅ Đã tìm thấy {len(valid_rejected_instances)} khách hàng bị TỪ CHỐI thực tế để test.")

    all_metrics = []
    success_count = 0

    for i, (x_req, original_prob) in enumerate(valid_rejected_instances):
        print(f"\n--- Đang xử lý Hồ sơ #{i+1}/{len(valid_rejected_instances)} (Prob gốc: {original_prob:.4f}) ---")
        
        # Sinh kịch bản Counterfactual (num_cf=3)
        cf_results = generator.generate(x_req, pop_size=100, n_gen=50, num_cf=3)
        
        if cf_results is None or cf_results.empty:
            print(" [!] Thất bại: Không tìm thấy CF hợp lệ.")
            continue
            
        print(f" [*] Thành công: Tìm thấy {len(cf_results)} CFs.")
        success_count += 1
        
        # Đánh giá các chỉ số chất lượng phản thực
        metrics = evaluator.evaluate(x_original=x_req, cf_df=cf_results)
        all_metrics.append(metrics)

    print("\n" + "="*50)
    print(f" KẾT QUẢ BENCHMARK TỔNG QUÁT ({dataset_name.upper()}) ")
    print("="*50)
    print(f"Tổng số hồ sơ đã test: {len(valid_rejected_instances)}")
    print(f"Số hồ sơ lật nhãn thành công (Success Rate): {success_count}/{len(valid_rejected_instances)} ({(success_count/len(valid_rejected_instances))*100:.2f}%)")

    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
            
        for key, value in avg_metrics.items():
            if key == "Total CFs Found":
                print(f" * {key} (Average per person): {value:.1f}")
            else:
                print(f" * {key}: {value:.4f}")
    else:
        print("Không có kết quả nào để đánh giá.")

if __name__ == "__main__":
    main()