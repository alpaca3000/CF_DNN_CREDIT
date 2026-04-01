from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from alibi.explainers import CounterfactualProto
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embed_mlp import EmbedMLP
from src.models.model_wrapper import EmbedMLPWrapper
from src.preprocess.preprocess import CreditPreprocessor
from src.preprocess.utils import load_data, split_data
from src.cf.metric_evaluator import CFMEvaluator
from src.cf.cfm_fm.generator import CFMFWGenerator

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

    if not isinstance(state, dict):
        raise ValueError("embed_mlp_best.pkl không phải state_dict như mong đợi.")

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, best_cfg


def _to_binary_probs(p1: np.ndarray) -> np.ndarray:
    p1 = np.asarray(p1, dtype=np.float32).reshape(-1)
    p0 = 1.0 - p1
    return np.column_stack([p0, p1])


def run_alibi_benchmark(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    wrapper: EmbedMLPWrapper,
    preprocessor: CreditPreprocessor,
    target_col: str = "Class",
    n_tests: int = 10,
    max_iterations: int = 500,
) -> None:
    print("\n" + "=" * 50)
    print(" CHẠY BENCHMARK VỚI ALIBI (CF-PROTO) ")
    print("=" * 50)

    # 1) Chuẩn bị dữ liệu train ở không gian scaled/encoded cho Alibi
    X_train_raw = df_train.drop(columns=[target_col])
    X_train_scaled = preprocessor.transform(X_train_raw).astype(np.float32)

    # 2) Hàm dự đoán cho Alibi (đầu vào là scaled/encoded)
    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        probs_1 = wrapper.predict_proba_encoded(x_np)
        return _to_binary_probs(probs_1)

    # 3) Khởi tạo và fit CF-Proto
    shape = (1, X_train_scaled.shape[1])
    cf_proto = CounterfactualProto(
        predict=predict_fn,       # ĐÃ SỬA: 'predict' thay vì 'predict_fn'
        shape=shape,
        use_kdtree=True,
        feature_range=(X_train_scaled.min(axis=0), X_train_scaled.max(axis=0)),
        max_iterations=max_iterations
        # ĐÃ XÓA: feature_names vì CounterfactualProto không hỗ trợ tham số này
    )

    print(" -> Đang huấn luyện prototypes trên tập Train...")
    cf_proto.fit(X_train_scaled)

    # 4) Khởi tạo evaluator giống CFM-FW (dùng cùng LOF module để chấm plausibility)
    generator = CFMFWGenerator(
        model_wrapper=wrapper,
        df_train_raw=df_train,
        target_col=target_col,
    )
    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=generator.plausibility_module,
        df_train_raw=df_train,
    )

    # 5) Lọc các hồ sơ thực sự bị từ chối
    potential_rejects = df_test[df_test[target_col] == 0].drop(columns=[target_col])
    valid_rejected_instances: list[tuple[pd.Series, float]] = []
    for _, row in potential_rejects.iterrows():
        prob = float(wrapper.predict_proba(row.to_frame().T)[0])
        if prob < 0.5:
            valid_rejected_instances.append((row, prob))
        if len(valid_rejected_instances) >= n_tests:
            break

    print(f"✅ Đã tìm thấy {len(valid_rejected_instances)} hồ sơ bị từ chối hợp lệ để test.")

    # 6) Sinh CF và chấm điểm
    all_metrics = []
    success_count = 0

    for i, (x_req, orig_prob) in enumerate(valid_rejected_instances, start=1):
        print(f"\n--- Hồ sơ #{i}/{len(valid_rejected_instances)} (Prob gốc: {orig_prob:.4f}) ---")

        x_req_scaled = preprocessor.transform(x_req.to_frame().T).astype(np.float32)
        explanation = cf_proto.explain(x_req_scaled, target_class=[1])

        if explanation.cf is None or explanation.cf.get("X", None) is None:
            print(" [!] Alibi thất bại: Không tìm thấy CF.")
            continue

        cf_scaled = np.asarray(explanation.cf["X"], dtype=np.float32)
        if cf_scaled.ndim == 1:
            cf_scaled = cf_scaled.reshape(1, -1)

        # QUAN TRỌNG: đưa từ dạng Scaled về dạng thô trước khi in/chấm sparsity
        cf_raw = preprocessor.inverse_transform(cf_scaled)
        cf_df = pd.DataFrame(cf_raw, columns=X_train_raw.columns)
        cf_df["predicted_prob"] = wrapper.predict_proba(cf_df)

        print(" [*] Alibi tìm thấy CF (raw):")
        print(cf_df)

        metrics = evaluator.evaluate(x_original=x_req, cf_df=cf_df)
        all_metrics.append(metrics)
        success_count += 1

    # 7) Tổng hợp kết quả
    print("\n" + "=" * 50)
    print(" KẾT QUẢ BENCHMARK ALIBI (AVERAGE METRICS) ")
    print("=" * 50)

    total = len(valid_rejected_instances)
    if total == 0:
        print("Không có hồ sơ phù hợp để benchmark.")
        return

    print(f"Tổng số hồ sơ đã test: {total}")
    print(f"Số hồ sơ lật nhãn thành công (Success Rate): {success_count}/{total} ({(success_count/total)*100:.2f}%)")

    if all_metrics:
        avg_metrics: dict[str, float] = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = float(sum(m[key] for m in all_metrics) / len(all_metrics))

        for key, value in avg_metrics.items():
            if key == "Total CFs Found":
                print(f" * {key} (Average per person): {value:.1f}")
            else:
                print(f" * {key}: {value:.4f}")
    else:
        print("Không có kết quả nào để đánh giá.")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ALIBI Counterfactual Generation")
    print("=" * 70)

    print("\n[STEP 1] Loading German Credit data...")
    df = load_data("german")
    print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")

    print("\n[STEP 2] Splitting data (80/20 train/test)...")
    df_train, df_test = split_data(df, "german", "Class", random_state=42)[:2]
    print(f"✅ Train: {len(df_train)}, Test: {len(df_test)}")

    X_train = df_train.drop("Class", axis=1)

    print("\n[STEP 3] Preprocessing with CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name="german", model_type="embedding")
    preprocessor.fit(X_train=X_train)
    metadata = preprocessor.get_metadata()
    print(f"✅ Num features: {len(metadata['num_features'])}")
    print(f"✅ Cat features: {len(metadata['cat_features'])}")

    print("\n[STEP 4] Load trained EmbedMLP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_cfg = _load_embed_model("germancredit", device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)
    print(f"✅ Loaded EmbedMLP with config: {best_cfg}")

    run_alibi_benchmark(
        df_train=df_train,
        df_test=df_test,
        wrapper=wrapper,
        preprocessor=preprocessor,
        target_col="Class",
        n_tests=10,
        max_iterations=100,
    )