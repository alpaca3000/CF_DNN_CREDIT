import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from typing import Any

# Allow running this file directly: `python testwraper.py`
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.embed_mlp import EmbedMLP

class EmbedMLPWrapper:
    """Wrapper dự đoán cho EmbedMLP, tích hợp sẵn preprocessing (scaler + ordinal encode)."""

    def __init__(self, model: EmbedMLP, preprocessor: Any, device: torch.device) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        metadata = self.preprocessor.get_metadata()
        self.feature_names = list(metadata["num_features"]) + list(metadata["cat_features"])
        self.num_features = metadata["num_features"]
        self.cat_features = metadata["cat_features"]
        self.cat_idxs = np.asarray(metadata["cat_idxs"], dtype=int)
        self.device = device

    def to_dataframe(self, X: pd.DataFrame | np.ndarray | dict[str, Any]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Thiếu cột đầu vào: {missing}")
            return X[self.feature_names].copy()

        if isinstance(X, dict):
            return pd.DataFrame([X], columns=self.feature_names)

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError("Số chiều đầu vào không khớp với feature_names của preprocessor.")
        return pd.DataFrame(X_arr, columns=self.feature_names)

    def transform(self, X: pd.DataFrame | np.ndarray | dict[str, Any]) -> np.ndarray:
        X_df = self.to_dataframe(X)
        # Đầu ra sau preprocessor phải giữ nguyên số cột (dùng OrdinalEncoder)
        # chuẩn hóa số liệu và mã hóa categorical bằng ordinal encoder (không one-hot)
        X_num = self.preprocessor.scaler.transform(X_df[self.preprocessor.num_features_])
        
        if self.preprocessor.cat_features_:
            # --- ĐOẠN SỬA ĐỔI: Ép kiểu string tuyệt đối chống lỗi so sánh hỗn hợp float/str ---
            cat_df = X_df[self.preprocessor.cat_features_].copy()
            for col in self.preprocessor.cat_features_:
                cat_df[col] = cat_df[col].fillna("MISSING").map(str).replace({"nan": "MISSING", "NaN": "MISSING"})
            
            X_cat = self.preprocessor.ordinal_encoder.transform(cat_df)
            encoded_data = np.hstack([X_num, X_cat])
        else:
            encoded_data = X_num
        return np.asarray(encoded_data, dtype=np.float32)

    def predict_proba_encoded(self, X_encoded: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_encoded, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Trích xuất đúng cột dựa trên index
        num_idxs = [i for i in range(X_arr.shape[1]) if i not in set(self.cat_idxs.tolist())]
        x_num = torch.from_numpy(X_arr[:, num_idxs]).float().to(self.device)

        if self.cat_idxs.size > 0:
            x_cat_arr = X_arr[:, self.cat_idxs].astype(np.int64)
            # Clip để đảm bảo ID không vượt quá số lượng categories khai báo
            for j, cdim in enumerate(self.model.cat_dims):
                x_cat_arr[:, j] = np.clip(x_cat_arr[:, j], 0, int(cdim) - 1)
            x_cat = torch.from_numpy(x_cat_arr).long().to(self.device)
        else:
            x_cat = torch.zeros((X_arr.shape[0], 0), dtype=torch.long, device=self.device)

        with torch.no_grad():
            probs = self.model(x_num, x_cat)
            
        # Dùng .flatten() để trả về mảng 1D [batch_size]
        return probs.detach().cpu().numpy().flatten()

    def predict_proba(self, X: pd.DataFrame | np.ndarray | dict[str, Any]) -> np.ndarray:
        if isinstance(X, dict):
            X = pd.DataFrame([X], columns=self.feature_names)
        X_encoded = self.transform(X)
        return self.predict_proba_encoded(X_encoded)