# usage: python -m src.cf.random_sample --dataset german_credit --n-tests 10 --seed 123 --n-runs 5 --fw cfm

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from dice_ml import Data, Dice, Model
import torch

import numpy as np
import pandas as pd
from src.models.model_wrapper import EmbedMLPWrapper
from src.cf.cfm_fm.lof import PlausibilityLOF
from src.models.loader import load_embed_model, _load_best_config, _load_decision_threshold
from src.data_processing.preprocess import CreditPreprocessor
from src.cf.metric_evaluator import CFMEvaluator
from src.data_processing.utils import load_data, split_data, get_target_col
from src.data_processing.domain_preprocess import GermanCreditDomainPreprocessor
from src.cf.dice_explainer import DiCEAdapter

# Setup đường dẫn hệ thống
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_SEEDS = [42, 456, 789, 1012, 1314]  # Danh sách các seed để chạy nhiều lần thực nghiệm

# tạo mẫu n_test với seed khác nhau từ tập dữ liệu test có cả feature và nhãn dự đoán
def random_sample(df: pd.DataFrame, target_col: str, n_test: int, seed: int = 123) -> pd.DataFrame:
    """
    Random sample n_test rows from df with a given seed.
    """
    if n_test is not None and n_test < len(df):
        df_subset = df.sample(n=n_test, random_state=seed)
    else:
        df_subset = df.copy()

    # Bỏ y_pred
    if "y_pred" in df_subset.columns:
        df_subset = df_subset.drop(columns=["y_pred"])
    # X_subset = df_subset.copy()
    return df_subset

def run_dice_benchmark(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    wrapper,
    preprocessor,
    target_col="target",
    decision_threshold: float = 0.5,
    num_cf=3,
    n_tests=10,
):

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


    # Truyền thêm cat_modes vào trong Adapter
    m = Model(model=DiCEAdapter(wrapper, d.feature_names, cat_modes), backend="sklearn")

    # 3. KHỞI TẠO DiCE EXPLAINER (Chuyển hẳn sang Random để dễ hội tụ với mạng MLP)
    print("[3] Khởi tạo DiCE Explainer (Random Search Algorithm)...")
    exp = Dice(d, m, method="random")

    # 4. KHỞI TẠO CFM EVALUATOR
    print("[4] Khởi tạo CFM Evaluator...")
    X_train_raw = df_train.drop(columns=[target_col])
    X_train_scaled = preprocessor.transform(X_train_raw)

    plausibility_module = PlausibilityLOF(n_neighbors=20,n_jobs=1)
    plausibility_module.fit(X_train_scaled)

    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=plausibility_module,
        df_train_raw=X_train_raw,
        decision_threshold=decision_threshold,
    )

    # 5. SINH CF & ĐÁNH GIÁ HÀNG LOẠT
    print("\n[5] Bắt đầu sinh CF bằng DiCE và Đánh giá...")

    # Tất cả các bộ dữ liệu đã quy ước chung: 0 là Bị từ chối
    test_sample_results = []
    success_count = 0

    for idx, row in df_test.iterrows():
        # Tách features ra khỏi target

        x_req_with_target = row.to_frame().T
        x_req = x_req_with_target.drop(columns=[target_col]).drop(columns=["y_prob"])
        

        print(f"--- Hồ sơ #{len(test_sample_results)+1} ---")

        try:
            dice_exp = exp.generate_counterfactuals(                                                    
                x_req, 
                total_CFs=num_cf, 
                desired_class=1,
                features_to_vary=actionable_features
            )

            dice_raw_df = dice_exp.cf_examples_list[0].final_cfs_df
            if dice_raw_df is not None and not dice_raw_df.empty:
                success_count += 1
                cf_df_clean = dice_raw_df.drop(columns=[target_col])

                # Điền cột xác suất dự đoán mới sau khi biến đổi qua DiCE
                cf_df_clean["predicted_prob"] = wrapper.predict_proba(cf_df_clean)

                metrics = evaluator.evaluate(x_original=row.drop(target_col), cf_df=cf_df_clean)
                test_sample_results.append(metrics)
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
        f"Số hồ sơ lật nhãn thành công (Success Rate): {success_count}/{len(test_sample_results) if test_sample_results else n_tests}"
    )

    if test_sample_results:
        avg_metrics = {}
        for key in test_sample_results[0].keys():
            avg_metrics[key] = sum(m[key] for m in test_sample_results) / len(test_sample_results)

        for key, value in avg_metrics.items():
            if key == "Total CFs Found":
                print(f" * {key}: {value:.1f}")
            else:
                print(f" * {key}: {value:.4f}")
    else:
        print("Không có kết quả đối chứng nào từ DiCE được ghi nhận.")
    return test_sample_results

import time
import pandas as pd
from src.cf.cfm_fm.generator import CFMFWGenerator
from src.cf.metric_evaluator import CFMEvaluator

def run_cfm_fw_benchmark(df_train_raw, df_subset, preprocessor, wrapper, decision_threshold, target_col, n_tests=100):
    print("\n" + "="*50)
    print(" BẮT ĐẦU CHẠY BENCHMARK: CFM-FW ")
    print("="*50)

    # 1. Khởi tạo Generator theo cấu trúc từ cfm_explainer
    generator = CFMFWGenerator(
        model_wrapper=wrapper,
        df_train_raw = df_train_raw,
        target_col=target_col
    )

    # Khởi tạo Evaluator (Lưu ý: X_train ở đây CHỈ chứa features, không chứa y_train)
    evaluator = CFMEvaluator(
        preprocessor=preprocessor,
        plausibility_module=generator.plausibility_module,
        df_train_raw=df_train_raw
    )

    all_metrics = []
    success_count = 0

    # 2. Vòng lặp sinh CF cho từng mẫu trong df_subset
    for idx, row in df_subset.iterrows():
        x_req = row.to_frame().T
        x_req = x_req.drop(columns=[target_col]).drop(columns=["y_prob"])  # Loại bỏ cột target và y_prob nếu có
        # kiểm tra nếu x_req không phải kiểu series thì chuyển thành kiểu series
        if not isinstance(x_req, pd.Series):
            x_req = x_req.squeeze()  # Chuyển DataFrame 1 dòng thành Series
        try:
            # Gọi hàm sinh CF từ generator
            cf_results = generator.generate(x_req, pop_size=100, n_gen=50, num_cf=3)

            if cf_results is None or cf_results.empty:
                print(f"[!] CFM-FW thất bại tại index {idx} (Không tìm thấy CF).")
                continue
                
            # Đánh giá các chỉ số
            metrics = evaluator.evaluate(x_original=x_req, cf_df=cf_results)
            all_metrics.append(metrics)
            success_count += 1
            print(f"[+] Index {idx}: Lật nhãn thành công.")

        except Exception as e:
            print(f"[-] Lỗi ngầm định tại index {idx}: {str(e)}")
            continue

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
    return all_metrics



def aggregate_and_report(fw_name: str, all_runs_data: list):
    """Tính toán Mean và Std từ mảng kết quả của nhiều Run và in ra Console."""
    print("\n" + "="*60)
    print(f" KẾT QUẢ THỰC NGHIỆM: {fw_name.upper()} ")
    print("="*60)
    
    if not all_runs_data or not all_runs_data[0]:
        print(" [!] Không có kết quả hợp lệ.")
        return

    metrics_keys = all_runs_data[0][0].keys()
    
    # Lấy trung bình từng vòng chạy
    run_averages = []
    for run_list in all_runs_data:
        if not run_list: continue
        avg_dict = {key: np.mean([m[key] for m in run_list if key in m]) for key in metrics_keys}
        run_averages.append(avg_dict)
        
    # Tính Mean ± Std tổng
    for key in metrics_keys:
        values = [r[key] for r in run_averages if key in r]
        if values:
            mean_val, std_val = np.mean(values), np.std(values)
            print(f" > {key:<25}: {mean_val:.4f} ± {std_val:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Random sample test data for counterfactual experiments.")
    parser.add_argument("--dataset", type=str, required=True, help="Tên bộ dữ liệu (ví dụ: 'german', 'taiwan', 'give_me_credit').")
    parser.add_argument("--n-tests", type=int, default=10, help="Số lượng mẫu test cần lấy ngẫu nhiên.")
    parser.add_argument("--seed", type=int, default=123, help="Seed để tái tạo mẫu ngẫu nhiên.")
    parser.add_argument("--fw", type=str, choices=["single_cf", "dice", "cfm", "all"], required=True)
    parser.add_argument("--n-runs", type=int, default=1, help="Số lần chạy thực nghiệm (Mặc định 1 lần để test)")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    n_test = args.n_tests
    current_seed = args.seed

    # Load data
    print(f"[INFO] Đang load dữ liệu cho bộ {dataset}...")
    df = load_data(dataset)
    target_col = get_target_col(dataset)

    print(f"[INFO] Đang chia dữ liệu thành train/test...")
    X_train, _, X_test, y_train, _, y_test = split_data(df, target_col)
    print(f"[INFO] Đã chia dữ liệu. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # --- SỬA LỖI 1: CHẠY DOMAIN PREP TRƯỚC KHI TẠO df_train_raw ---
    print(f"[INFO] Đang tiền xử lý dữ liệu (Domain Preprocess)...")
    if dataset == "german_credit":
        domain_prep = GermanCreditDomainPreprocessor()
        X_train = domain_prep.transform(X_train)
        X_test  = domain_prep.transform(X_test)
    elif dataset == "gmsc":
        pass

    # df_train_raw giờ đã khớp hoàn toàn số chiều với tập Test
    df_train_raw = pd.concat([X_train, y_train], axis=1)

    preprocessor = CreditPreprocessor(dataset)
    preprocessor.fit(X_train)
    X_test_processed = preprocessor.transform(X_test)
    y_test_processed = y_test.to_numpy().astype(int)
    metadata = preprocessor.get_metadata()

    # --- SỬA LỖI 2: CHUẨN BỊ DỮ LIỆU CHUỖI RIÊNG CHO DICE ---
    X_train_dice = X_train.copy()
    for col in metadata["cat_features"]:
        if col in X_train_dice.columns:
            X_train_dice[col] = X_train_dice[col].fillna("MISSING").astype(str)
    df_train_dice = pd.concat([X_train_dice, y_train], axis=1)

    print(f"[INFO] Đang load EmbedMLP model...")
    model = load_embed_model(dataset, device)
    threshold = _load_decision_threshold(dataset)
    wrapper = EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device)

    # Predict on test set
    cat_idxs = metadata.get("cat_idxs", [])
    from src.models.trainer import predict
    from src.models.loader import create_embedding_dataloaders
    test_loader = create_embedding_dataloaders(X=X_test_processed, y=y_test_processed, cat_idxs=cat_idxs, shuffle=False)
    y_prob, y_pred = predict(model=model, test_loader=test_loader, device=device, threshold=threshold)

    y_prob_numpy = y_prob.detach().cpu().numpy().reshape(-1)
    y_pred_numpy = y_pred.detach().cpu().numpy().reshape(-1)
    y_true_numpy = np.asarray(y_test_processed).reshape(-1)

    # --- TẠO TẬP HỒ SƠ TỪ CHỐI BẰNG DỮ LIỆU NGUYÊN BẢN (SẠCH) ---
    df_test_clean = pd.concat([X_test, y_test], axis=1)
    df_test_clean["y_pred"] = y_pred.detach().cpu().numpy().reshape(-1)
    df_test_clean["y_prob"] = y_prob.detach().cpu().numpy().reshape(-1)

    # Filter rejected cases (predicteed label = 0 or probability < threshold)
    print(f"[INFO] Đang lọc các hồ sơ bị từ chối (y_pred=0)...")
    df_rejected = df_test_clean[df_test_clean["y_pred"] == 0].copy() # X, label, y_pred, y_prob

    results_store = {fw: [] for fw in ["single_cf", "dice", "cfm"]}
    target_fws = ["single_cf", "dice", "cfm"] if args.fw == "all" else [args.fw]
    runs = min(args.n_runs, len(EXPERIMENT_SEEDS))

    # loop 5 times
    for run_idx in range(runs):
        print(f"\n[INFO] Lần chạy thứ {run_idx+1} với seed = {EXPERIMENT_SEEDS[run_idx]}...")
        df_sampled_i = random_sample(df_rejected, target_col, n_test, EXPERIMENT_SEEDS[run_idx]) # must be X, label, y_prod
        current_seed = EXPERIMENT_SEEDS[run_idx]
        np.random.seed(current_seed)

        if "dice" in target_fws:
            print("\n[INFO] Đang chạy DiCE...")

            df_sampled_dice = df_sampled_i.copy()
            for col in metadata["cat_features"]:
                if col in df_sampled_dice.columns:
                    df_sampled_dice[col] = df_sampled_dice[col].fillna("MISSING").astype(str)

            result_sample_dice = run_dice_benchmark(
                df_train=df_train_dice, 
                df_test=df_sampled_dice, 
                preprocessor=preprocessor,
                wrapper=wrapper,
                target_col=target_col,
                num_cf=3,
                decision_threshold=threshold
                # Lưu ý: Cập nhật hàm run_dice_benchmark của bạn để nhận thêm biến desired_class nếu cần
            )
            results_store["dice"].append(result_sample_dice)

        if "cfm" in target_fws:
            print("\n[INFO] Đang chạy CFM-FW...")
            res_cfm = run_cfm_fw_benchmark(
                df_train_raw=df_train_raw, 
                df_subset=df_sampled_i, # Tập mẫu test đã chọn qua hàm sample
                preprocessor=preprocessor,
                wrapper=wrapper,
                decision_threshold=threshold,
                target_col=target_col,
                n_tests=n_test
            )
            results_store["cfm"].append(res_cfm)
            pass

        if "single_cf" in target_fws:
            print("\n[INFO] Đang chạy Single-CF...")
            # res_single_cf = run_single_cf_benchmark(df_train=X_train,
            #                                         df_subset=df_sampled_i,
            #                                         preprocessor=preprocessor,
            #                                         wrapper=EmbedMLPWrapper(model=model, preprocessor=preprocessor, device=device),
            #                                         num_cf=3,
            #                                         decision_threshold=threshold
            # )
            # results_store["single_cf"].append(res_single_cf)
            pass

    # 5. In kết quả Console
    print(f"\n[DEBUG] Số lượng vòng chạy thực tế được lưu vào mảng: {len(results_store['dice'])}")
    for fw in target_fws:
        aggregate_and_report(fw, results_store[fw])



if __name__ == "__main__":
    main()     
     