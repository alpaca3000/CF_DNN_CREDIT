from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from typing import Any
import json
from sklearn.model_selection import train_test_split


# Allow running this file directly: `python testwraper.py`
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cf.cfm_fm.core.acme_weight import AcMEEngine
from src.models.embed_mlp import EmbedMLP
from src.cf.cfm_fm.core.model_wrapper import EmbedMLPWrapper
from src.preprocess.preprocess import CreditPreprocessor
from src.cf.cfm_fm.core.kdtree_population import KDTreeInitializer, KDTreeMixedSampling
from src.cf.cfm_fm.core.evaluator import CFMEvaluator
from src.cf.cfm_fm.core.generator import CFMFWGenerator

OUTPUTS_DIR = PROJECT_ROOT / "src" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"


def _normalize_class_labels(class_series: pd.Series) -> pd.Series:
    # Ưu tiên xử lý numeric labels trước
    numeric = pd.to_numeric(class_series, errors="coerce")
    if numeric.notna().all():
        numeric = numeric.astype(int)
        uniq = set(numeric.unique().tolist())
        if uniq.issubset({0, 1}):
            return numeric.astype(int)
        if uniq.issubset({1, 2}):
            return numeric.map({1: 1, 2: 0}).astype(int)

    # Xử lý string labels (good/bad, approved/rejected...)
    text = class_series.astype(str).str.strip().str.lower()
    text_map = {
        "1": 1,
        "0": 0,
        "2": 0,
        "good": 1,
        "bad": 0,
        "approved": 1,
        "rejected": 0,
        "true": 1,
        "false": 0,
    }
    mapped = text.map(text_map)
    if mapped.notna().all():
        return mapped.astype(int)

    raise ValueError(
        f"Không chuẩn hóa được nhãn Class về 0/1. Giá trị đang có: {sorted(text.unique().tolist())}"
    )

def _split_raw_train(
    df: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> pd.DataFrame:
    X_trainval_df, _, y_trainval, _ = train_test_split(
        df,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    val_ratio_in_trainval = val_size / (1.0 - test_size)
    X_train_df, _, y_train, _ = train_test_split(
        X_trainval_df,
        y_trainval,
        test_size=val_ratio_in_trainval,
        random_state=seed,
        stratify=y_trainval,
    )
    return X_train_df.copy(), y_train.copy()

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
    # load data
    print("\n[1/5] Load dữ liệu và Preprocessor...")
    dataset = pd.read_csv(PROJECT_ROOT / "data/german_credit.csv")
    print("1. Đã load dữ liệu thành công. Dữ liệu có dạng:")
    print(dataset.head(5))

    dataset["Class"] = _normalize_class_labels(dataset["Class"])

    y = dataset["Class"].values
    X = dataset.drop(columns=["Class"])
    feature_names = X.columns.tolist()

    X_train_raw, y_train_raw = _split_raw_train(X, y, seed=42)
    df_train_raw = X_train_raw.copy()
    df_train_raw["Class"] = y_train_raw

    print("\n2. Đang huấn luyện (fit) CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name="german", model_type="embedding")
    preprocessor.fit(X_train_raw)

    metadata = preprocessor.get_metadata()
    print(f"\n=> CreditPreprocessor đã fit xong. Metadata thu được:")
    print(f"   - num_features: {metadata['num_features']}")
    print(f"   - cat_features: {metadata['cat_features']}")
    print(f"   - cat_dims: {metadata['cat_dims']}")

    # 2. KHỞI TẠO MÔ HÌNH Pytorch (Wrapper)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_num_dim = len(metadata["num_features"])
    cat_dims = metadata["cat_dims"]
    model, best_cfg = _load_embed_model("germancredit", device)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)

    # 3. LẤY MỘT KHÁCH HÀNG BỊ TỪ CHỐI ĐỂ TEST
    print("\n[3/5] Trích xuất khách hàng cần giải thích...")
    rejected_pool = df_train_raw[df_train_raw["Class"] == 0]
    if rejected_pool.empty:
        raise ValueError("Không tìm thấy khách hàng bị từ chối (Class = 0) trong tập train sau khi split.")

    x_rejected = rejected_pool.iloc[0].drop("Class")
    print("Hồ sơ gốc (Bị từ chối):")
    print(x_rejected.to_frame().T.to_string(index=False))

    # double-check dự đoán của model wrapper cho hồ sơ gốc
    prob_rejected = wrapper.predict_proba(x_rejected.to_frame().T)[0]
    print(f"\nXác suất duyệt vay theo model wrapper: {prob_rejected:.4f} (Phải < 0.5 để đảm bảo đây là hồ sơ bị từ chối thực sự)")
    assert prob_rejected < 0.5, "Lỗi: Hồ sơ gốc không bị từ chối theo model wrapper. Vui lòng kiểm tra lại dữ liệu và model."

    # 4. CHẠY GENERATOR (Trái tim của hệ thống)
    print("\n[4/5] Kích hoạt CFM-FW Generator (AcME + KDTree + NSGA-II)...")
    generator = CFMFWGenerator(model_wrapper=wrapper, df_train=df_train_raw, preprocessor=preprocessor)

    # Sinh CF (Đặt pop_size nhỏ lại để test chạy nhanh)
    cf_results = generator.generate(x_rejected, pop_size=40, n_gen=30, k_neighbors=5)
    
    # 5. ĐÁNH GIÁ VÀ XUẤT BÁO CÁO (Evaluator)
    print("\n[5/5] Đánh giá chất lượng toán học của các giải pháp...")
    evaluator = CFMEvaluator(
        preprocessor=preprocessor, 
        df_train=df_train_raw,
        feature_weights=generator.acme.explain(df_train_raw.drop(columns=['Class']))["abs_weights"]
    )

    metrics = evaluator.evaluate(x_original=x_rejected, cf_df=cf_results)

    # =====================================================================
    # XUẤT BÁO CÁO CUỐI CÙNG (DÙNG ĐỂ CHỤP ẢNH BỎ VÀO KHÓA LUẬN)
    # =====================================================================
    print("\n" + "="*60)
    print(" KẾT QUẢ ĐỀ XUẤT PHẢN THỰC TẾ (TOP 3)")
    print("="*60)
    if not cf_results.empty:
        # Sắp xếp ưu tiên theo xác suất và hiển thị Top 3
        top_cfs = cf_results.sort_values(by='predicted_prob', ascending=False).head(3)
        print(top_cfs.to_string(index=False))
        
        print("\n" + "-"*60)
        print(" BẢNG BENCHMARK (ĐO LƯỜNG CHỈ SỐ)")
        print("-"*60)
        for metric, value in metrics.items():
            print(f"{metric:<35}: {value}")
            
        print("\n=> KẾT LUẬN: Hệ thống đã áp dụng thành công Ràng buộc Nhân quả, "
              "đồng thời thay đổi Thu nhập/Khoản vay để đạt xác suất duyệt an toàn (>0.5).")
    else:
        print("Thuật toán không tìm được giải pháp nào hợp lệ trong không gian tìm kiếm hiện tại.")    


    # ### TEST ACME ENGINE
    # acme = AcMEEngine.from_dataframe_baseline(
    #     predict_proba=wrapper.predict_proba,
    #     df=X_train_raw_df,
    #     num_features=metadata["num_features"],
    #     cat_features=metadata["cat_features"]
    # )

    # print("\n4. [Baseline Profile] - Hồ sơ trung bình/phổ biến nhất:")  
    # print(acme.baseline_series)

    # print("\nĐang tính toán ma trận hiệu ứng (có thể mất vài giây nếu mô hình lớn)...")
    # explanation = acme.explain(X_train_raw_df)

    # print("\n[Ma trận hiệu ứng] - 5 dòng đầu tiên:")
    # print(explanation["effects"].head())

    # print("\n[KẾT QUẢ QUAN TRỌNG] - Trọng số đặc trưng (Đã chuẩn hóa tổng = 1):")
    # weights = explanation["abs_weights"]
    # print(weights.sort_values(ascending=False))
    
    # assert not weights.isna().any(), "Lỗi: Có giá trị NaN trong trọng số."
    # print("\n=> CHÚC MỪNG! AcME Engine đã trích xuất trọng số thành công.")


    ### TEST THỬ MODEL WRAPPER RIÊNG BIỆT (KHÔNG QUA ACME) ĐỂ ĐẢM BẢO TÍNH ĐÚNG ĐẮN CỦA WRAPPER TRƯỚC KHI DÙNG CHO ACME
    # x_test_df = X_train_raw_df.iloc[[1]].copy() # Lấy khách hàng số 2
    # prob_1 = wrapper.predict_proba(x_test_df)
    # print(f"\n[Test 1] Dự đoán khách hàng DataFrame: Xác suất duyệt vay = {prob_1[0]:.4f}")

    # # 3.5 Test suy luận với định dạng Dict (NSGA-II sẽ truyền vào dạng này)
    # # đọc khách hàn số 10 rồi chuyên thành dict
    # x_test_dict = X_train_raw_df.iloc[9].to_dict()
    # prob_2 = wrapper.predict_proba(x_test_dict)
    # print(f"[Test 2] Dự đoán khách hàng Dictionary: Xác suất duyệt vay = {prob_2[0]:.4f}")

    # # 3.6 Kiểm tra tính hợp lệ
    # assert isinstance(prob_1, np.ndarray), "Lỗi: Đầu ra không phải là numpy array"
    # assert prob_1.ndim == 1, "Lỗi: Đầu ra chưa được làm phẳng thành 1D array"
    # print("\n=> CHÚC MỪNG! Pipeline Preprocessor + Wrapper đã hoạt động hoàn hảo!")

    # ### TEST KDTREE
    # # 3. Test KDTreeInitializer
    # print("\nKhởi tạo KDTree với dữ liệu đã được duyệt vay (Nhãn = 1)...")
    # # gộp X_train_raw và y_train_raw thành một DataFrame duy nhất để KDTreeInitializer dễ dàng xử lý
    # X_train_raw_df = X_train_raw.copy()
    # X_train_raw_df["Class"] = y_train_raw
    # kdtree_engine = KDTreeInitializer(
    #     df_train=X_train_raw_df, 
    #     preprocessor=preprocessor, 
    #     target_col="Class", 
    #     good_label=1
    # )
    
    # # Khách hàng bị từ chối (Lấy dòng index 1, target = 0)
    # x_rejected = X_train_raw.iloc[[1]].copy()
    # print("\n[Hồ sơ bị từ chối (Gốc)]:")
    # print(x_rejected.to_dict(orient='records')[0])
    
    # # Lấy 2 láng giềng tốt nhất
    # print("\nTìm kiếm 2 láng giềng gần nhất đã được duyệt vay...")
    # kdtree_pop = kdtree_engine.get_initial_population(x_rejected, k=2)
    
    # for i, record in enumerate(kdtree_pop):
    #     print(f"Láng giềng {i+1}: {record}")
        
    # # 4. Test KDTreeMixedSampling (Giả lập pymoo gọi hàm)
    # print("\nKiểm tra lớp bọc KDTreeMixedSampling cho pymoo...")
    # sampling = KDTreeMixedSampling(kdtree_pop)
    
    # # Giả sử pymoo muốn khởi tạo quần thể gồm 5 cá thể (pop_size = 5)
    # # Vì KDTree chỉ tìm được 2 láng giềng, Sampling phải lặp lại dữ liệu để điền đủ 5
    # mock_population = sampling._do(problem=None, n_samples=5)
    
    # print(f"Kích thước quần thể sinh ra: {mock_population.shape}")
    # print(f"Kiểu dữ liệu của mảng: {mock_population.dtype} (Phải là 'object')")
    # print("Phần tử đầu tiên trong quần thể:", mock_population[0])
    # print("Phần tử thứ 3 (phải lặp lại từ đầu):", mock_population[2])
    
    # # Assertions
    # assert isinstance(kdtree_pop, list), "Lỗi: Đầu ra KDTree không phải là list."
    # assert mock_population.dtype == object, "Lỗi: Mảng Sampling không phải là dtype=object."
    # assert mock_population.shape[0] == 5, "Lỗi: Sampling không sinh đủ số lượng cá thể yêu cầu."
    
    # print("\n=> CHÚC MỪNG! Module KDTree và Sampling đã hoạt động hoàn hảo.")