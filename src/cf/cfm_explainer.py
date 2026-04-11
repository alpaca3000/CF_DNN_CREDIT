# usage: python -m src.cf.cfm_explainer

from pathlib import Path
import sys
import torch
from typing import Any
import json

from src.models.embed_mlp import EmbedMLP
from src.models.model_wrapper import EmbedMLPWrapper
from src.preprocess.preprocess import CreditPreprocessor
from src.cf.metric_evaluator import CFMEvaluator
from src.cf.cfm_fm.generator import CFMFWGenerator
from src.data_processing.utils import load_data, split_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    df = load_data('german_credit')
    print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")

    print("\n[STEP 2] Splitting data (80/20 train/test)...")
    split_result = split_data(df, 'german_credit', 'Class', random_state=42)
    df_train, df_test = split_result[0], split_result[1]
    print(f"✅ Train: {len(df_train)}, Test: {len(df_test)}")

    X_train = df_train.drop('Class', axis=1)
    y_train = df_train['Class']
    X_test = df_test.drop('Class', axis=1)
    y_test = df_test['Class']

    print("\n[STEP 3] Preprocessing with CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name='german_credit', model_type='embedding')
    preprocessor.fit(X_train=X_train)
    metadata = preprocessor.get_metadata()
    print(f"✅ Num features: {len(metadata['num_features'])}")
    print(f"✅ Cat features: {len(metadata['cat_features'])}")

    print("\n[STEP 4] Load trained EmbedMLP model...")
    # load file models .pklfrom MODELS_DIR/german_credit/embed_mlp_best.pkl hoặc lấy best config trong results/german_credit/best_configs.json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_cfg = _load_embed_model('german_credit', device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)
    print(f"✅ Loaded EmbedMLP with config: {best_cfg}")

    print("\n[STEP 5] Initialize CFM-FM Generator...")
    generator = CFMFWGenerator(
        model_wrapper=wrapper,
        df_train_raw=df_train,
        target_col='Class'
    )
    print("✅ CFM-FM Generator initialized.")

    print("\n[STEP 6] & [STEP 7] BATCH EVALUATION - Khởi tạo Đánh giá hàng loạt...")
    
    # Khởi tạo Evaluator 1 lần duy nhất ở ngoài vòng lặp
    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=generator.plausibility_module,
        df_train_raw=df_train
    )

    # 1. Lọc ra danh sách khách hàng thực sự bị TỪ CHỐI bởi mô hình trong tập TEST
    N_TESTS = 10  # Số lượng mẫu bạn muốn test (có thể tăng lên 50, 100)
    valid_rejected_instances = []
    
    # Chỉ lấy những người có Class = 0 thực tế
    potential_rejects = df_test[df_test["Class"] == 0].drop(columns=["Class"])
    
    for _, row in potential_rejects.iterrows():
        # Double check: Mô hình phải thực sự từ chối người này (< 0.5)
        prob = wrapper.predict_proba(row.to_frame().T)[0]
        if prob < 0.5:
            valid_rejected_instances.append((row, prob))
        
        if len(valid_rejected_instances) == N_TESTS:
            break

    print(f"✅ Đã tìm thấy {len(valid_rejected_instances)} khách hàng bị từ chối hợp lệ để test.")

    # 2. Chạy vòng lặp sinh CF và Đánh giá
    all_metrics = []
    success_count = 0

    for i, (x_req, original_prob) in enumerate(valid_rejected_instances):
        print(f"\n--- Đang xử lý Hồ sơ #{i+1}/{len(valid_rejected_instances)} (Prob gốc: {original_prob:.4f}) ---")
        
        # Sinh kịch bản CF (Có thể giới hạn num_cf=3 để chạy nhanh hơn và sát thực tế)
        cf_results = generator.generate(x_req, pop_size=100, n_gen=50, num_cf=3)
        
        if cf_results is None or cf_results.empty:
            print(" [!] Thất bại: Không tìm thấy CF hợp lệ.")
            continue
            
        print(f" [*] Thành công: Tìm thấy {len(cf_results)} CFs.")
        success_count += 1
        
        # Đánh giá CF cho riêng khách hàng này
        metrics = evaluator.evaluate(x_original=x_req, cf_df=cf_results)
        all_metrics.append(metrics)

    # 3. Tổng hợp kết quả (Average Benchmark)
    print("\n" + "="*50)
    print(" KẾT QUẢ BENCHMARK TỔNG QUÁT (AVERAGE METRICS) ")
    print("="*50)
    
    print(f"Tổng số hồ sơ đã test: {len(valid_rejected_instances)}")
    print(f"Số hồ sơ lật nhãn thành công (Success Rate): {success_count}/{len(valid_rejected_instances)} ({(success_count/len(valid_rejected_instances))*100:.2f}%)")

    if all_metrics:
        # Tính trung bình các value trong list dictionary
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
    