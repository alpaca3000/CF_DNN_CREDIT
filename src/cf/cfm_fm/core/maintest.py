import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Allow running this file directly: `python testwraper.py`
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from src.cf.cfm_fm.core.evaluator import plausibility_score
from src.preprocess.preprocess import CreditPreprocessor

# Giả sử bạn đã có:
# - preprocessor: bộ tiền xử lý đã fit() ở bước trước
# - df_train: tập dữ liệu huấn luyện
# - df_test: tập dữ liệu kiểm thử (chứa dữ liệu thật chưa từng thấy)
# - plausibility_score: hàm tính LOF-NR từ bài báo

def sanity_check_lof(df_train: pd.DataFrame, df_test: pd.DataFrame, preprocessor, target_col='target'):
    print("--- BẮT ĐẦU KIỂM CHỨNG HÀM LOF-NR ---")
    
    # 1. Tập Reference (Fit LOF): Lấy những người Tốt từ tập Train
    df_train_good = df_train[df_train[target_col] == 1].drop(columns=[target_col])
    X_ref_encoded = preprocessor.transform(df_train_good)
    
    # 2. Tập Test (Chấm điểm LOF): Lấy những người Tốt thật sự từ tập Test
    df_test_good = df_test[df_test[target_col] == 1].drop(columns=[target_col])
    X_test_encoded = preprocessor.transform(df_test_good)
    
    print(f"Số lượng hồ sơ tham chiếu (Train Good): {len(X_ref_encoded)}")
    print(f"Số lượng hồ sơ kiểm tra (Test Good): {len(X_test_encoded)}")
    
    # 3. Chấm điểm LOF cho những người tốt trong tập Test
    # Về lý thuyết, vì họ là người tốt thực tế, LOF phải nhận diện họ là "bình thường"
    score = plausibility_score(
        X_cf=X_test_encoded, 
        X_ref=X_ref_encoded, 
        n_neighbors=20 # Theo bài báo thường dùng 20
    )
    
    print(f"\n=> Điểm LOF-NR của NHỮNG NGƯỜI TỐT TRONG THỰC TẾ: {score:.4f}")
    
    if score < 0.5:
        print("[KẾT LUẬN]: Hàm LOF đang có vấn đề! Nó đang đánh giá chính người thật là Outlier.")
    else:
        print("[KẾT LUẬN]: Hàm LOF hoạt động bình thường! Lỗi do NSGA-II sinh ra CF quá ảo.")

# Gọi hàm chạy thử
# chia tập train test cho bộ dữ liệu german credit


dataset = pd.read_csv(PROJECT_ROOT / "data/german_credit.csv")
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['Class'])

# Fit the preprocessor on the training data
preprocessor = CreditPreprocessor()
preprocessor.fit(train_df.drop(columns=['Class']))

sanity_check_lof(train_df, test_df, preprocessor, target_col='Class')