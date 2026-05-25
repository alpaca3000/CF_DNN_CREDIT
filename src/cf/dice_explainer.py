# usage:
#   python -m src.cf.dice_explainer --dataset german_credit --n-tests 10
#   python -m src.cf.dice_explainer --dataset lending_club --n-tests 10
#   python -m src.cf.dice_explainer --dataset gmsc --n-tests 10

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from dice_ml import Data, Dice, Model

# Đảm bảo PROJECT_ROOT nằm trong sys.path để tránh lỗi ModuleNotFoundError
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cf.cfm_fm.lof import PlausibilityLOF
from src.cf.metric_evaluator import CFMEvaluator
from src.data_processing.preprocess import CreditPreprocessor
from src.data_processing.utils import get_target_col, load_data, split_data
from src.models.embed_mlp import EmbedMLP
from src.models.model_wrapper import EmbedMLPWrapper

OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"


def run_dice_benchmark(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    wrapper,
    preprocessor,
    target_col="target",
    num_cf=3,
    n_tests=10,
):
    print("\n" + "=" * 60)
    print(f" CHẠY BENCHMARK VỚI DiCE (BASELINE) - {target_col.upper()} ")
    print("=" * 60)

    metadata = preprocessor.get_metadata()
    num_features = metadata["num_features"]
    actionable_features = metadata["actionable"]

    # 1. KHỞI TẠO DiCE DATA
    print("[1] Khởi tạo DiCE Data...")
    d = Data(dataframe=df_train, continuous_features=num_features, outcome_name=target_col)

    # 2. KHỞI TẠO DiCE MODEL (Adapter)
    print("[2] Khởi tạo DiCE Model (Bọc Black-box Wrapper)...")

    # Lấy giá trị xuất hiện nhiều nhất (mode) của các cột để làm sạch dữ liệu đột biến của DiCE
    cat_modes = {col: df_train[col].mode()[0] for col in metadata["cat_features"] if col in df_train.columns}

    class DiCEAdapter:
        def __init__(self, wrapper_instance, feature_names, cat_modes_dict):
            self.wrapper = wrapper_instance
            self.feature_names = feature_names
            self.cat_modes = cat_modes_dict
            
        def predict_proba(self, X):
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_names)
            
            X_clean = X.copy()
            # Bẫy lỗi cưỡng bức: Thay thế tất cả các biến thể "MISSING", "nan" hoặc rỗng
            # bằng giá trị hợp lệ (mode) để tránh kích hoạt lỗi Categorical của DiCE
            for col, mode_val in self.cat_modes.items():
                if col in X_clean.columns:
                    # Nếu xuất hiện giá trị lạ không thuộc danh mục, hoặc rỗng, đưa về mode
                    X_clean[col] = X_clean[col].astype(str).str.strip()
                    invalid_mask = X_clean[col].isin(["MISSING", "nan", "NaN", ""])
                    if invalid_mask.any():
                        X_clean.loc[invalid_mask, col] = str(mode_val)
            
            probs = self.wrapper.predict_proba(X_clean)
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

    # Truyền thêm cat_modes vào trong Adapter
    m = Model(model=DiCEAdapter(wrapper, d.feature_names, cat_modes), backend="sklearn")

    # 3. KHỞI TẠO DiCE EXPLAINER (Chuyển hẳn sang Random để dễ hội tụ với mạng MLP)
    print("[3] Khởi tạo DiCE Explainer (Random Search Algorithm)...")
    exp = Dice(d, m, method="random")

    # 4. KHỞI TẠO CFM EVALUATOR (Dùng chung thước đo gắt gao của đề tài)
    print("[4] Khởi tạo CFM Evaluator...")
    X_train_raw = df_train.drop(columns=[target_col])
    X_train_scaled = preprocessor.transform(X_train_raw)

    plausibility_module = PlausibilityLOF(n_neighbors=20)
    plausibility_module.fit(X_train_scaled)

    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=plausibility_module,
        df_train_raw=X_train_raw,
    )

    # 5. SINH CF & ĐÁNH GIÁ HÀNG LOẠT
    print("\n[5] Bắt đầu sinh CF bằng DiCE và Đánh giá...")

    # Tất cả các bộ dữ liệu đã quy ước chung: 0 là Bị từ chối
    potential_rejects = df_test[df_test[target_col] == 0]

    all_metrics = []
    success_count = 0

    for idx, row in potential_rejects.iterrows():
        if len(all_metrics) >= n_tests:
            break

        # Tách features ra khỏi target
        x_req_with_target = row.to_frame().T
        x_req = x_req_with_target.drop(columns=[target_col])

        # Tính xác suất được duyệt thực tế từ mô hình gốc
        prob = wrapper.predict_proba(x_req)[0]

        # Nếu mô hình đoán xác suất duyệt >= 0.5, nghĩa là không bị từ chối thực tế -> Bỏ qua
        if prob >= 0.5:
            continue

        print(f"--- Hồ sơ #{len(all_metrics)+1} (Prob gốc: {prob:.4f}) ---")

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

                # Điền cột xác suất dự đoán mới sau khi biến đổi qua DiCE
                cf_df_clean["predicted_prob"] = wrapper.predict_proba(cf_df_clean)

                metrics = evaluator.evaluate(x_original=row.drop(target_col), cf_df=cf_df_clean)
                all_metrics.append(metrics)
            else:
                print(" [!] DiCE không tìm thấy CF.")
        except Exception as e:
            print(f" [!] Lỗi khi sinh bằng DiCE: {e}")

    # 6. IN KẾT QUẢ BENCHMARK TỔNG QUÁT
    print("\n" + "=" * 60)
    print(f" KẾT QUẢ BENCHMARK DiCE (BASELINE FOR {target_col.upper()}) ")
    print("=" * 60)
    print(f"Tổng số hồ sơ mục tiêu: {n_tests}")
    print(
        f"Số hồ sơ lật nhãn thành công (Success Rate): {success_count}/{len(all_metrics) if all_metrics else n_tests}"
    )

    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        for key, value in avg_metrics.items():
            if key == "Total CFs Found":
                print(f" * {key}: {value:.1f}")
            else:
                print(f" * {key}: {value:.4f}")
    else:
        print("Không có kết quả đối chứng nào từ DiCE được ghi nhận.")


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
    parser = argparse.ArgumentParser(description="Batch Evaluation for DiCE Baseline.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german_credit",
        choices=["german_credit", "lending_club", "gmsc"],
        help="Dataset name to evaluate.",
    )
    parser.add_argument("--n-tests", type=int, default=10, help="Number of rejected profiles to test.")
    args = parser.parse_args()

    dataset_name = args.dataset
    target_col = get_target_col(dataset_name)

    print("=" * 70)
    print(f"Running DiCE Explainer Baseline on: {dataset_name.upper()}")
    print("=" * 70)

    # [STEP 1] Load data
    print(f"\n[STEP 1] Loading {dataset_name} data...")
    df = load_data(dataset_name)
    print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")

    # [STEP 2] Splitting data
    print("\n[STEP 2] Splitting data...")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df=df, target_col=target_col
    )

    # [STEP 3] Preprocessing (ĐƯỢC ĐẨY LÊN TRƯỚC ĐỂ LẤY KHỞI TẠO PREPROCESSOR)
    print("\n[STEP 3] Preprocessing with CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name=dataset_name, model_type="embedding")
    preprocessor.fit(X_train=X_train)
    metadata = preprocessor.get_metadata()
    print(f"✅ Num features: {len(metadata['num_features'])}")
    print(f"✅ Cat features: {len(metadata['cat_features'])}")

    # --- ĐOẠN ĐÃ SỬA: Ép kiểu string an toàn cho DiCE sau khi preprocessor đã khởi tạo ---
    X_train_dice = X_train.copy()
    X_test_dice = X_test.copy()
    
    for col in preprocessor.cat_features_:
        if col in X_train_dice.columns:
            X_train_dice[col] = X_train_dice[col].fillna("MISSING").astype(str)
        if col in X_test_dice.columns:
            X_test_dice[col] = X_test_dice[col].fillna("MISSING").astype(str)

    # Tái cấu trúc DataFrame gộp nhãn phục vụ thư viện DiCE
    df_train = pd.concat([X_train_dice, y_train], axis=1)
    df_test = pd.concat([X_test_dice, y_test], axis=1)
    print(f"✅ Train features: {X_train.shape}, Test features: {X_test.shape}")

    # [STEP 4] Load trained model wrapper
    print("\n[STEP 4] Load trained EmbedMLP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_cfg = _load_embed_model(dataset_name, device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)
    print(f"✅ Loaded EmbedMLP model wrapper.")

    # Run Benchmark
    run_dice_benchmark(
        df_train=df_train,
        df_test=df_test,
        wrapper=wrapper,
        preprocessor=preprocessor,
        target_col=target_col,
        num_cf=3,
        n_tests=args.n_tests,
    )


if __name__ == "__main__":
    main()