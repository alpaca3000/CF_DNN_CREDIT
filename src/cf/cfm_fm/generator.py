from pathlib import Path
import sys
from typing import Any

import pandas as pd
import numpy as np

# Allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_wrapper import EmbedMLPWrapper
from src.cf.cfm_fm.acme_weight import AcMEEngine
from src.cf.cfm_fm.kdtree_population import KDTreeInitializer, KDTreeMixedSampling
from src.cf.cfm_fm.lof import PlausibilityLOF
from src.cf.cfm_fm.problem import CFMProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.optimize import minimize

class CFMFWGenerator:
    """
    Bộ điều phối chính thức của khung CFM-FW.
    Kết nối BlackBox, AcME, KDTree, PlausibilityLOF và NSGA-II.
    """
    def __init__(self, model_wrapper: Any, df_train_raw: pd.DataFrame, target_col: str = 'Class'):
        self.wrapper = model_wrapper
        self.df_train_raw = df_train_raw.copy()
        self.target_col = target_col
        
        # Preprocessor lấy từ wrapper để đảm bảo tính đồng nhất
        self.preprocessor = self.wrapper.preprocessor 
        self.metadata = self.preprocessor.get_metadata()
        
        print("[Generator] Khởi tạo hệ thống Offline Setup...")
        
        # =================================================================
        # 1. Khởi tạo KDTree (Tìm kiếm Prototypes - Những hồ sơ được duyệt)
        # =================================================================
        print(" -> Xây dựng KDTree từ tập dữ liệu Inliers (Được duyệt vay)...")
        self.kdtree = KDTreeInitializer(
            df_train=self.df_train_raw, 
            preprocessor=self.preprocessor, 
            target_col=self.target_col, 
            good_label=1
        )
        
        # Dữ liệu Train thô (bỏ cột Target) để phục vụ AcME và Problem
        self.X_train_raw = self.df_train_raw.drop(columns=[self.target_col])
        
        # =================================================================
        # 2. Huấn luyện Plausibility (LOF)
        # =================================================================
        print(" -> Huấn luyện mô hình Local Outlier Factor (LOF)...")
        # Scale dữ liệu train trước khi đưa vào LOF
        X_train_scaled = self.preprocessor.transform(self.X_train_raw)
        self.plausibility_module = PlausibilityLOF(n_neighbors=20)
        self.plausibility_module.fit(X_train_scaled)
        
        # =================================================================
        # 3. Tính toán Trọng số Đặc trưng (AcME)
        # =================================================================
        print(" -> Chạy AcME Engine để trích xuất Feature Weights...")
        self.acme = AcMEEngine.from_dataframe_baseline(
            predict_proba=self.wrapper.predict_proba,
            df=self.X_train_raw,
            num_features=self.metadata["num_features"],
            cat_features=self.metadata["cat_features"],
            target_class=1
        )
        explanation = self.acme.explain(self.X_train_raw)
        self.feature_weights = explanation["weights"] # Đã sửa key
        
        print("[Generator] Khởi tạo thành công. Sẵn sàng sinh Phản thực tế!")

    def generate(
        self, 
        x_rejected: pd.Series, 
        num_cf: int = 5,
        pop_size: int = 50, 
        n_gen: int = 50, 
        k_neighbors: int = 50
    ) -> pd.DataFrame:
        """
        Sinh các mẫu Counterfactual cho một khách hàng bị từ chối.
        """
        # =========================================================
        # PHA 1: Khởi tạo hạt giống (Seed Population) bằng KDTree
        # =========================================================
        # Chuyển x_rejected thành DataFrame 1 dòng
        x_req_df = x_rejected.to_frame().T
        
        # Lấy các mẫu đã đậu (prototypes) gần nhất
        kdtree_pop_df = self.kdtree.get_initial_population(x_req_df, k=k_neighbors)
        
        # Nếu kdtree_pop_df là DataFrame, lấy list of dicts. 
        # Nếu đã là list of dicts thì dùng trực tiếp (Tùy thuộc implementation của KDTreeMixedSampling)
        pop_records = kdtree_pop_df.to_dict('records') if isinstance(kdtree_pop_df, pd.DataFrame) else kdtree_pop_df
        custom_sampling = KDTreeMixedSampling(pop_records)

        # =========================================================
        # PHA 2: Thiết lập Bài toán Đa mục tiêu (CFMProblem)
        # =========================================================
        problem = CFMProblem(
            model_wrapper=self.wrapper,
            original_instance=x_rejected,
            feature_weights=self.feature_weights,
            metadata=self.metadata,
            df_train=self.X_train_raw,               # Truyền df thô để tính ranges
            plausibility_module=self.plausibility_module, # Truyền thẳng class LOF mới
            target_prob=0.51
        )

        # =========================================================
        # PHA 3: Cấu hình Thuật toán NSGA-II
        # =========================================================
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=custom_sampling,
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )

        # =========================================================
        # PHA 4: Tiến hóa (Optimization Loop)
        # =========================================================
        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', n_gen),
            seed=42,
            verbose=False
        )

        # =========================================================
        # PHA 5: Trích xuất và Lọc Kết quả (Pareto Front)
        # =========================================================
        if res.X is None:
            return pd.DataFrame()

        # res.X của MixedVariable thường là np.array chứa các dictionary
        solutions = [sol for sol in res.X] if isinstance(res.X, np.ndarray) else [res.X]
        df_cf = pd.DataFrame(solutions)
        
        # Kiểm tra tính hợp lệ bằng Batch Prediction (Nhanh hơn)
        preds = self.wrapper.predict_proba(df_cf)
        
        # Xử lý nếu pred là 1D hay 2D array
        if preds.ndim == 1:
            df_cf['predicted_prob'] = preds
        else:
            df_cf['predicted_prob'] = preds[:, 1] # Lấy xác suất của class 1
            
        # Lọc những hồ sơ vượt ngưỡng
        df_valid_cf = df_cf[df_cf['predicted_prob'] >= 0.50].reset_index(drop=True)
        
        x_orig_dict = x_rejected.to_dict()
        all_features = self.metadata["num_features"] + self.metadata["cat_features"]
        
        def count_changes(row):
            changes = 0
            for col in all_features:
                # Dùng np.isclose cho số thực để tránh sai số dấu phẩy động
                if isinstance(row[col], (int, float)) and isinstance(x_orig_dict[col], (int, float)):
                    if not np.isclose(row[col], x_orig_dict[col], atol=1e-5):
                        changes += 1
                elif row[col] != x_orig_dict[col]:
                    changes += 1
            return changes

        # Thêm cột tính điểm số thay đổi
        df_valid_cf['n_changes'] = df_valid_cf.apply(count_changes, axis=1)
        
        # 3. Sắp xếp: Ưu tiên thay đổi ÍT NHẤT (Sparsity), 
        # Nếu số lượng thay đổi bằng nhau, ưu tiên mẫu có xác suất an toàn cao hơn.
        df_valid_cf = df_valid_cf.sort_values(
            by=['n_changes', 'predicted_prob'], 
            ascending=[True, False]
        )
        
        # 4. Cắt lấy số lượng num_cf theo yêu cầu và dọn dẹp cột phụ
        df_final = df_valid_cf.head(num_cf).drop(columns=['n_changes']).reset_index(drop=True)
        
        return df_final


# ==========================================
# TEST SCRIPT: CFMFWGenerator on German Credit
# ==========================================
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from src.preprocess.preprocess import CreditPreprocessor
    from src.preprocess.utils import load_data, split_data
    from src.cf.cfm_fm.lof import PlausibilityLOF
    
    print("=" * 70)
    print("[CFMFWGenerator TEST] German Credit Dataset")
    print("=" * 70)
    
    # ============ STEP 1: Load & Preprocess ============
    print("\n[STEP 1] Loading German Credit data...")
    df = load_data('german')
    print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")
    
    print("\n[STEP 2] Splitting data (80/20 train/test)...")
    split_result = split_data(df, 'german', 'Class', random_state=42)
    df_train, df_test = split_result[0], split_result[1]
    print(f"✅ Train: {len(df_train)}, Test: {len(df_test)}")
    
    print("\n[STEP 3] Preprocessing with CreditPreprocessor...")
    preprocessor = CreditPreprocessor(dataset_name='german', model_type='embedding')
    preprocessor.fit(df_train.drop('Class', axis=1))
    metadata = preprocessor.get_metadata()
    print(f"✅ Num features: {len(metadata['num_features'])}")
    print(f"✅ Cat features: {len(metadata['cat_features'])}")
    
    # ============ STEP 4: Train Model ============
    print("\n[STEP 4] Training RandomForest model...")
    X_train = preprocessor.transform(df_train.drop('Class', axis=1))
    y_train = df_train['Class'].values.astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"✅ Model trained. Train accuracy: {model.score(X_train, y_train):.4f}")
    
    # ============ STEP 5: Create Model Wrapper ============
    print("\n[STEP 5] Creating ModelWrapper...")
    
    class ModelWrapper:
        def __init__(self, model, preprocessor):
            self.model = model
            self.preprocessor = preprocessor
        
        def predict_proba(self, x_df):
            x_transformed = self.preprocessor.transform(x_df)
            probs = self.model.predict_proba(x_transformed)
            return probs
    
    model_wrapper = ModelWrapper(model, preprocessor)
    print("✅ ModelWrapper ready")
    
    # ============ STEP 6: Initialize CFMFWGenerator ============
    print("\n[STEP 6] Initializing CFMFWGenerator...")
    generator = CFMFWGenerator(
        model_wrapper=model_wrapper,
        df_train_raw=df_train,
        target_col='Class'
    )
    print("✅ CFMFWGenerator initialized")
    
    # ============ STEP 7: Select 3 Test Samples ============
    print("\n[STEP 7] Selecting 3 test samples (class 0 predicted)...")
    X_test_raw = df_test.drop('Class', axis=1).copy()
    y_test = df_test['Class'].values
    test_probs = model_wrapper.predict_proba(X_test_raw)
    prob_class1_all = test_probs[:, 1] if test_probs.ndim == 2 else test_probs
    idx_candidates = np.where(prob_class1_all < 0.51)[0]
    
    if len(idx_candidates) < 3:
        print(f"⚠️  Only {len(idx_candidates)} samples with pred class 0, using first 3 available")
        test_indices = list(idx_candidates) + list(range(max(len(idx_candidates), 3)))[:3-len(idx_candidates)]
    else:
        test_indices = list(idx_candidates[:3])
    
    print(f"✅ Selected indices: {test_indices}")
    
    # ============ STEP 8: Generate CFs ============
    print("\n" + "=" * 70)
    print("GENERATING COUNTERFACTUALS FOR 3 TEST SAMPLES")
    print("=" * 70)
    
    all_cf_results = []
    
    for sample_idx, idx in enumerate(test_indices, 1):
        print(f"\n{'─' * 70}")
        print(f"[SAMPLE {sample_idx}/3] Index: {idx}")
        print(f"{'─' * 70}")
        
        x_rejected = X_test_raw.iloc[idx]
        actual_class = int(y_test[idx])
        pred_prob = float(prob_class1_all[idx])
        pred_class = 1 if pred_prob >= 0.51 else 0
        
        print(f"Original state:")
        print(f"  Actual class:        {actual_class}")
        print(f"  Predicted prob:      {pred_prob:.4f}")
        print(f"  Predicted class:     {pred_class}")
        
        # Kiểm tra xem mẫu có thực sự là class 0 không
        if pred_class != 0:
            print(f"⚠️  SKIP: Original sample is NOT class 0 (pred={pred_prob:.4f} >= 0.51)")
            all_cf_results.append({
                'sample_idx': idx,
                'actual_class': actual_class,
                'original_prob': pred_prob,
                'original_class': pred_class,
                'n_generated': 0,
                'n_flipped': 0,
                'flip_rate': 0.0,
                'cf_mean_prob': None,
                'status': 'SKIPPED'
            })
            continue
        
        print(f"\nGenerating CFs (pop_size=40, n_gen=30)...")
        try:
            df_cf = generator.generate(
                x_rejected=x_rejected,
                pop_size=40,
                n_gen=30,
                k_neighbors=50
            )
            
            n_generated = len(df_cf)
            print(f"✅ Generated {n_generated} valid CFs (prob_class1 ≥ 0.50)")
            
            if n_generated > 0:
                # Kiểm tra xem bao nhiêu CF thực sự lật nhãn từ 0 -> 1
                probs_cf = df_cf['predicted_prob'].values
                n_flipped = np.sum(probs_cf >= 0.51)
                flip_rate = n_flipped / n_generated
                
                print(f"\nFlip Analysis (CRITICAL):")
                print(f"  ❌ Original:      class 0 (prob={pred_prob:.4f})")
                print(f"  ✅ Target:       class 1 (prob ≥ 0.51)")
                print(f"  📊 CFs flipped:  {n_flipped}/{n_generated} ({flip_rate:.1%})")
                
                if n_flipped == 0:
                    print(f"  ⚠️  CRITICAL: NO CFs flipped the label!")
                
                print(f"\nCF Statistics (prob_class1):")
                print(f"  Min:    {probs_cf.min():.4f}")
                print(f"  Max:    {probs_cf.max():.4f}")
                print(f"  Mean:   {probs_cf.mean():.4f}")
                print(f"  Median: {np.median(probs_cf):.4f}")
                
                # Top 3 best CFs (highest prob)
                if n_flipped > 0:
                    top_indices = np.argsort(probs_cf)[-3:][::-1]
                    print(f"\nTop 3 CFs (by prob_class1):")
                    for rank, cf_idx in enumerate(top_indices, 1):
                        prob_val = probs_cf[cf_idx]
                        status = "✅" if prob_val >= 0.51 else "❌"
                        print(f"  [{rank}] prob={prob_val:.4f} {status}")
                
                status = 'SUCCESS' if n_flipped > 0 else 'FAIL'
                all_cf_results.append({
                    'sample_idx': idx,
                    'actual_class': actual_class,
                    'original_prob': pred_prob,
                    'original_class': pred_class,
                    'n_generated': n_generated,
                    'n_flipped': n_flipped,
                    'flip_rate': flip_rate,
                    'cf_mean_prob': probs_cf.mean(),
                    'status': status
                })
            else:
                print("❌ FAIL: No valid CFs generated (prob_class1 < 0.50)")
                all_cf_results.append({
                    'sample_idx': idx,
                    'actual_class': actual_class,
                    'original_prob': pred_prob,
                    'original_class': pred_class,
                    'n_generated': 0,
                    'n_flipped': 0,
                    'flip_rate': 0.0,
                    'cf_mean_prob': None,
                    'status': 'FAIL'
                })
        
        except Exception as e:
            print(f"❌ Error generating CFs: {str(e)}")
            all_cf_results.append({
                'sample_idx': idx,
                'actual_class': actual_class,
                'original_prob': pred_prob,
                'original_class': pred_class,
                'n_generated': -1,
                'n_flipped': -1,
                'flip_rate': -1.0,
                'cf_mean_prob': None,
                'status': 'ERROR'
            })
    
    # ============ SUMMARY ============
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    summary_df = pd.DataFrame(all_cf_results)
    print("\n" + summary_df.to_string(index=False))
    
    successful = (summary_df['status'] == 'SUCCESS').sum()
    total_flipped = summary_df['n_flipped'].sum()
    total_generated = summary_df[summary_df['n_generated'] > 0]['n_generated'].sum()
    
    print(f"\n{'─' * 70}")
    print(f"Samples with successful label flip: {successful}/{len(summary_df)}")
    print(f"Total CFs generated: {total_generated}")
    print(f"Total CFs with label flip: {total_flipped}")
    if total_generated > 0:
        overall_flip_rate = total_flipped / total_generated
        print(f"Overall flip rate: {overall_flip_rate:.1%}")
    print(f"{'─' * 70}")
    
    if successful == len(summary_df):
        print("\n✅ [SUCCESS] All 3 samples achieved label flip!")
    elif successful > 0:
        print(f"\n⚠️  [PARTIAL] Only {successful}/3 samples achieved label flip")
    else:
        print("\n❌ [FAILURE] No samples achieved label flip")
    
    print("=" * 70)