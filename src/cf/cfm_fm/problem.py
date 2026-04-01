from typing import Dict, Any, Sequence
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice
import numpy as np
import sys
from pathlib import Path
# Allow running this file directly: `python testwraper.py`
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.cf.cfm_fm.objectives import validity, proximity, rim, plausibility


def create_pymoo_space(df_train: pd.DataFrame, metadata: dict, x_original: pd.Series) -> dict:
    space = {}
    num_features = metadata["num_features"]
    cat_features = metadata["cat_features"]
    immutable = metadata.get("immutable", [])
    
    for col in num_features:
        if col in immutable:
            val = float(x_original[col])
            # Khóa cứng biến không đổi
            space[col] = Real(bounds=(val, val))
        else:
            # Lấy min/max thực tế từ tập train để đảm bảo tính thực tế
            min_val, max_val = float(df_train[col].min()), float(df_train[col].max())
            space[col] = Real(bounds=(min_val, max_val))
            
    for col in cat_features:
        if col in immutable:
            space[col] = Choice(options=[x_original[col]])
        else:
            unique_vals = df_train[col].dropna().unique().tolist()
            space[col] = Choice(options=unique_vals)
            
    return space

class CFMProblem(ElementwiseProblem):
    """
    Bài toán tối ưu hóa đa mục tiêu Pymoo cho Counterfactual Explanation.
    """
    def __init__(
        self,
        model_wrapper: Any,
        original_instance: pd.Series,
        feature_weights: pd.Series,
        metadata: Dict[str, Any],
        df_train: pd.DataFrame,          
        plausibility_module: Any,        # Nhận trực tiếp class PlausibilityLOF
        target_prob: float = 0.51,
        **kwargs
    ):
        self.model_wrapper = model_wrapper
        self.original_series = original_instance
        self.original_dict = original_instance.to_dict()
        
        # Metadata
        self.num_features = metadata["num_features"]
        self.cat_features = metadata["cat_features"]
        self.all_features = self.num_features + self.cat_features
        self.cat_mask = np.array([col in self.cat_features for col in self.all_features])
        self.target_prob = target_prob
        
        num_ranges_dict = metadata.get("num_ranges_dict", {})
        self.num_ranges = np.array([num_ranges_dict.get(col, 1.0) for col in self.num_features])
        
        # Sắp xếp trọng số AcME theo đúng thứ tự all_features
        self.w_arr = np.array([feature_weights.get(col, 1.0) for col in self.all_features])
        
        # LOF components
        self.plausibility_module = plausibility_module
        self.lof_stats = plausibility_module.get_train_stats()

        # Khởi tạo G (ieq_constr) dựa trên causal_rules
        self.causal_rules = metadata.get("causal_rules", [])
        
        # Khởi tạo không gian tìm kiếm
        self.space = create_pymoo_space(df_train, metadata, original_instance)
        
        # Trong pymoo mixed variables, phải truyền tham số vars
        super().__init__(
            vars=self.space,
            n_obj=4, 
            n_ieq_constr=len(self.causal_rules), 
            **kwargs
        )

    def _dict_to_array(self, x_dict: Dict[str, Any]) -> np.ndarray:
        """Chuyển dict cá thể thành array theo thứ tự all_features"""
        return np.array([x_dict[col] for col in self.all_features])

    def _evaluate(self, x: Dict[str, Any], out: Dict[str, Any], *args, **kwargs):
        # 1. Chuyển đổi định dạng dữ liệu
        x_arr = self._dict_to_array(x)
        orig_arr = self._dict_to_array(self.original_dict)
        
        # Tạo DataFrame 1 dòng dùng cho các model dự đoán
        x_df = pd.DataFrame([x])

        # --- F1: VALIDITY ---
        # model_wrapper nhận DataFrame thô, tự động pre-process và dự đoán
        pred_probs = self.model_wrapper.predict_proba(x_df)
        
        # Xử lý nếu predict_proba trả về mảng 1D hoặc 2D
        if pred_probs.ndim == 1:
            prob = pred_probs[0] if pred_probs.shape[0] == 1 else pred_probs[1]
        else:
            prob = pred_probs[0, 1] # Giả sử mục tiêu là class 1
            
        f1_validity = validity(prob, self.target_prob)

        # --- F2: PROXIMITY ---
        f2_proximity = proximity(
            orig_arr, x_arr, self.w_arr, self.num_ranges, cat_mask=self.cat_mask
        )

        # --- F3: RIM ---
        f3_rim = rim(
            orig_arr, x_arr, self.w_arr, self.num_ranges, cat_mask=self.cat_mask
        )

        # --- F4: PLAUSIBILITY ---
        # Giả sử model_wrapper có bộ preprocessor để dùng chung
        x_scaled = self.model_wrapper.preprocessor.transform(x_df)
        current_lof_score = self.plausibility_module.score_samples(x_scaled)[0]
        
        f4_plausibility = plausibility(
            current_lof_score, 
            self.lof_stats['min'], 
            self.lof_stats['max']
        )

        # --- G: CAUSAL CONSTRAINTS ---
        # Constraint G <= 0
        G = []
        for rule in self.causal_rules:
            col = rule['feature']
            if rule['type'] == '>=':    # Tuổi mới >= Tuổi cũ -> Tuổi cũ - Tuổi mới <= 0
                G.append(self.original_dict[col] - x[col])
            elif rule['type'] == '<=':  # Mới <= Cũ -> Mới - Cũ <= 0
                G.append(x[col] - self.original_dict[col])
            elif rule['type'] == '==':  # Mới == Cũ
                G.append(1.0 if x[col] != self.original_dict[col] else -1.0)

        # --- OUTPUT ---
        out["F"] = [f1_validity, f2_proximity, f3_rim, f4_plausibility]
        if G: 
            out["G"] = G


# # Model wrapper để kết nối với CreditPreprocessor và model
# class ModelWrapper:
#     """Wrapper cho model đã được train kèm CreditPreprocessor"""
#     def __init__(self, model: Any, preprocessor: Any) -> None:
#         self.model = model
#         self.preprocessor = preprocessor
    
#     def predict_proba(self, x_df: pd.DataFrame) -> Any:
#         """Tiền xử lý + dự đoán xác suất"""
#         x_transformed = self.preprocessor.transform(x_df)
#         probs = self.model.predict_proba(x_transformed)
#         return probs


# # ==========================================
# # PIPELINE: GERMAN CREDIT DATASET
# # ==========================================
# if __name__ == "__main__":
#     from sklearn.ensemble import RandomForestClassifier
#     from src.preprocess.preprocess import CreditPreprocessor
#     from src.preprocess.utils import load_data, split_data
#     from src.cf.cfm_fm.lof import PlausibilityLOF
#     from src.cf.cfm_fm.acme_weight import AcMEEngine
    
#     print("=" * 70)
#     print("[1/6] Loading German Credit data...")
#     print("=" * 70)
#     df = load_data('german')
#     print(f"✅ Loaded {len(df)} records, {len(df.columns)} features")
    
#     print("\n" + "=" * 70)
#     print("[2/6] Splitting data (80/20 train/test)...")
#     print("=" * 70)
#     split_result = split_data(df, 'german', 'Class', random_state=42)
#     df_train, df_test = split_result[0], split_result[1]
#     print(f"✅ Train: {len(df_train)}, Test: {len(df_test)}")
    
#     print("\n" + "=" * 70)
#     print("[3/6] Preprocessing with CreditPreprocessor...")
#     print("=" * 70)
#     preprocessor = CreditPreprocessor(dataset_name='german', model_type='embedding')
#     preprocessor.fit(df_train.drop('Class', axis=1))
    
#     # Lấy metadata để xác định features
#     metadata = preprocessor.get_metadata()
#     print(f"✅ Num features: {len(metadata['num_features'])}")
#     print(f"✅ Cat features: {len(metadata['cat_features'])}")
    
#     # Transform train data
#     X_train = preprocessor.transform(df_train.drop('Class', axis=1))
#     y_train = df_train['Class'].values.astype(int)
    
#     print("\n" + "=" * 70)
#     print("[4/6] Training RandomForest model...")
#     print("=" * 70)
#     model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
#     model.fit(X_train, y_train)
#     print(f"✅ Model trained. Train accuracy: {model.score(X_train, y_train):.4f}")
    
#     # Tạo model wrapper
#     model_wrapper = ModelWrapper(model, preprocessor)
    
#     print("\n" + "=" * 70)
#     print("[5/6] Computing AcME feature weights...")
#     print("=" * 70)
#     X_train_raw = df_train.drop('Class', axis=1)
#     num_features = [c for c in metadata['num_features'] if c in X_train_raw.columns]
#     cat_features = [c for c in metadata['cat_features'] if c in X_train_raw.columns]
    
#     acme = AcMEEngine.from_dataframe_baseline(
#         predict_proba=model_wrapper.predict_proba,
#         df=X_train_raw,
#         num_features=num_features,
#         cat_features=cat_features,
#         target_class=1
#     )
    
#     explanation = acme.explain(X_train_raw, quantiles=(0.25, 0.50, 0.75))
#     weights = explanation['weights']
    
#     print(f"✅ Weights computed. Top 5:")
#     print(weights.nlargest(5))
    
#     print("\n" + "=" * 70)
#     print("[6/6] Fitting PlausibilityLOF...")
#     print("=" * 70)
#     X_train_transformed = preprocessor.transform(df_train.drop('Class', axis=1))
#     plausibility_lof = PlausibilityLOF(contamination=0.1)
#     plausibility_lof.fit(X_train_transformed)
#     lof_stats = plausibility_lof.get_train_stats()
#     print(f"✅ LOF fitted. Min score: {lof_stats['min']:.4f}, Max: {lof_stats['max']:.4f}")
    
#     print("\n" + "=" * 70)
#     print("[7/7] Creating CFMProblem and running tests...")
#     print("=" * 70)
    
#     # Chọn sample test có xác suất class 1 thấp hơn ngưỡng để test lật nhãn 0 -> 1
#     X_test_raw = df_test.drop('Class', axis=1).copy()
#     y_test = df_test['Class'].values
#     test_probs = model_wrapper.predict_proba(X_test_raw)
#     prob_class1_all = test_probs[:, 1] if test_probs.ndim == 2 else test_probs
#     idx_candidates = np.where(prob_class1_all < 0.51)[0]

#     if len(idx_candidates) > 0:
#         idx_test = int(idx_candidates[0])
#     else:
#         idx_test = 0

#     x_test = X_test_raw.iloc[idx_test]
#     actual_class = int(y_test[idx_test])
#     original_prob_class1 = float(prob_class1_all[idx_test])
#     original_pred_class = int(original_prob_class1 >= 0.51)
    
#     print(
#         f"\nSelected test instance (index {idx_test}, actual class={actual_class}, "
#         f"pred class={original_pred_class}, prob_class1={original_prob_class1:.4f}):"
#     )
#     print(x_test)
    
#     # Tạo CFMProblem
#     metadata_with_rules = metadata.copy()
#     metadata_with_rules['causal_rules'] = [
#         # Năm tuổi thường không giảm (age non-decreasing)
#         {'feature': 'Age', 'type': '>=', 'ref': 'original'},
#     ]
    
#     problem = CFMProblem(
#         model_wrapper=model_wrapper,
#         original_instance=x_test,
#         feature_weights=weights,
#         metadata=metadata_with_rules,
#         df_train=df_train.drop('Class', axis=1),
#         plausibility_module=plausibility_lof,
#         target_prob=0.51
#     )
    
#     print("\n" + "=" * 70)
#     print("KỊCH BẢN TEST CFMProblem")
#     print("=" * 70)
    
#     def run_test(name: str, candidate_dict: Dict[str, Any]) -> None:
#         """Chạy test với một candidate counterfactual"""
#         out: Dict[str, Any] = {}
#         problem._evaluate(candidate_dict, out)
        
#         print(f"\n>>> TEST CASE: {name}")
#         print(f"Candidate changes from original:")
#         for k, v in candidate_dict.items():
#             if k in x_test.index and candidate_dict[k] != x_test[k]:
#                 print(f"  {k}: {x_test[k]} → {v}")
        
#         f1, f2, f3, f4 = out['F']
        
#         # Dự đoán cho candidate
#         x_cand_df = pd.DataFrame([candidate_dict])
#         pred_proba = model_wrapper.predict_proba(x_cand_df)
#         pred_proba_class1 = pred_proba[0, 1] if pred_proba.ndim == 2 else pred_proba[1]
        
#         print(f"\nObjective Scores:")
#         print(f"  F1 Validity    = {f1:.6f} (prob_class1={pred_proba_class1:.4f}, threshold=0.51)")
#         print(f"  F2 Proximity   = {f2:.6f} (weighted distance)")
#         print(f"  F3 RIM         = {f3:.6f} (relative impact measure)")
#         print(f"  F4 Plausibility= {f4:.6f} (LOF-based outlier score)")
#         print(
#             f"  Flip status    = {'✅ flipped to class 1' if pred_proba_class1 >= 0.51 else '❌ still class 0'}"
#         )
        
#         if f1 == 0:
#             print("✅ Validity: ACHIEVED (crossed decision boundary)")
#         else:
#             print(f"❌ Validity: Not achieved yet (loss={f1:.6f})")
        
#         G = out.get('G', [])
#         if G:
#             print(f"\nConstraints (must be ≤ 0 to satisfy):")
#             for i, g_val in enumerate(G):
#                 status = "✅" if g_val <= 0 else "❌"
#                 print(f"  G[{i}] = {g_val:.6f} {status}")
    
#     # Test 1: Original instance (baseline)
#     run_test("Baseline (Original)", x_test.to_dict())
    
#     # Test 2: Perturb numeric features to improve creditworthiness
#     candidate_2 = x_test.to_dict()
#     if 'Age' in candidate_2:
#         candidate_2['Age'] = min(float(x_test['Age']) + 1.0, float(df_train['Age'].max()))
#     if 'Status' in candidate_2: # chọn status khác nếu có
#         candidate_2['Status'] = '>= 200 DM / salary assignment >= 1 year'
#     run_test("Improve Creditworthiness (Modify numeric)", candidate_2)
    
#     # Test 3: Candidate that violates causal rule
#     candidate_3 = x_test.to_dict()
#     if 'Age' in candidate_3:
#         candidate_3['Age'] = max(float(x_test['Age']) - 1.0, float(df_train['Age'].min()))
#     run_test("Causal Rule Violation (Reduce age)", candidate_3)
    
#     print("\n" + "=" * 70)
#     print("[SUCCESS] CFMProblem test script completed with German Credit data!")
#     print("=" * 70)