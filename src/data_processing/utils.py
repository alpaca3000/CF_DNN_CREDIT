from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing.preprocess import CreditPreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_data(dataset_name: str) -> pd.DataFrame:
    """Load dữ liệu từ data/ theo dataset_name đã chuẩn xác."""
    file_map = {
        "german_credit": "german_credit.csv",
        "gmsc": "gmsc.csv",
        "lending_club": "lendingclub.csv",
    }

    if dataset_name not in file_map:
        raise ValueError("dataset_name phải là: 'german_credit', 'gmsc', hoặc 'lending_club'.")

    path = DATA_DIR / file_map[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    return pd.read_csv(path)

def split_data(
    df: pd.DataFrame,
    target_col: str,             # Đổi tên từ output_col thành target_col để đồng bộ
    dataset_name: str = None,    # Thêm tham số này để tránh lỗi khi truyền 'german_credit' vào vị trí thứ 2
    val_size: float = 0.2,       # Đổi default sang fraction cho chuẩn sklearn
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Tách dữ liệu thành X_train, X_valid, X_test, y_train, y_valid, y_test.
    """
    # Kiểm tra khi truyền nhầm dataset_name vào vị trí target_col
    # Ví dụ: split_data(df, 'german_credit', 'Class') -> Lúc này Class sẽ bị đẩy vào dataset_name
    if target_col not in df.columns:
        # Thử đảo ngược nếu có dataset_name hợp lệ nằm ở vị trí target_col
        if dataset_name in df.columns:
            target_col, dataset_name = dataset_name, target_col
        else:
            raise ValueError(f"Không tìm thấy cột mục tiêu '{target_col}' trong dataframe.")

    val_ratio = val_size / 100 if val_size > 1 else val_size
    test_ratio = test_size / 100 if test_size > 1 else test_size
    
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("test_size/val_size phải nằm trong khoảng (0, 1) hoặc là phần trăm dương.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratify giúp giữ nguyên tỷ lệ các lớp (Duyệt/Từ chối)
    stratify_y = y if y.nunique() > 1 else None

    # Tách Test set
    X_trainval, X_test, y_trainval, y_test = train_test_split( 
        X, y,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify_y,
    )

    # Tách Validation set từ phần còn lại
    trainval_stratify = y_trainval if y_trainval.nunique() > 1 else None
    val_ratio_in_trainval = val_ratio / (1.0 - test_ratio)
    
    X_train, X_valid, y_train, y_valid = train_test_split(  
        X_trainval, y_trainval,
        test_size=val_ratio_in_trainval,
        random_state=random_state,
        stratify=trainval_stratify,
    )

    return (
        X_train.reset_index(drop=True),  
        X_valid.reset_index(drop=True),  
        X_test.reset_index(drop=True),  
        y_train.reset_index(drop=True),  
        y_valid.reset_index(drop=True),  
        y_test.reset_index(drop=True),  
    )

def get_target_col(dataset_name: str) -> str:
    """Trả về tên cột nhãn dựa theo tập dữ liệu."""
    target_map = {
        "german_credit": "Class",
        "gmsc": "SeriousDlqin2yrs",
        "lending_club": "target",
    }
    if dataset_name not in target_map:
        raise ValueError(f"Không nhận diện được dataset: {dataset_name}")
    return target_map[dataset_name]


# def preprocess_data(
# 	X_train: pd.DataFrame,
# 	X_valid: pd.DataFrame,
# 	X_test: pd.DataFrame,
# 	dataset_name: str,
# 	model: str,
# ) -> Dict[str, Any]:
# 	"""
# 	Preprocess data cho model type được chỉ định.

# 	Args:
# 		X_train, X_valid, X_test: DataFrames chứa features (không có target).
# 		dataset_name: tên dataset ("german_credit", "gmsc", "lending_club").
# 		model: loại model ("classic_mlp", "embed_mlp", "xgboost", "random_forest").

# 	Returns:
# 		Dictionary chứa:
# 		- "preprocessor": CreditPreprocessor instance
# 		- "metadata": dict với cat_idxs, cat_dims, actionable, immutable, etc.
# 		- "X_train_pp", "X_valid_pp", "X_test_pp": transformed arrays
# 	"""

# 	print("\n" + "="*60)
# 	print(f"PREPROCESSING FOR {model.upper()}")
# 	print("="*60)

# 	# Map model names to CreditPreprocessor model_type
# 	model_type_map = {
# 		"classic_mlp": "classic_mlp",
# 		"embed_mlp": "embedding",
# 		"xgboost": "tree",
# 		"random_forest": "tree",
# 	}

# 	if model not in model_type_map:
# 		raise ValueError(f"Unknown model: {model}. Must be one of {list(model_type_map.keys())}")

# 	model_type = model_type_map[model]

# 	pp = CreditPreprocessor(dataset_name=dataset_name, model_type=model_type)
# 	pp.fit(X_train)
# 	metadata = pp.get_metadata()

# 	return {
# 		"preprocessor": pp,
# 		"metadata": metadata,
# 		"X_train_pp": pp.transform(X_train),
# 		"X_valid_pp": pp.transform(X_valid),
# 		"X_test_pp": pp.transform(X_test),
# 	}