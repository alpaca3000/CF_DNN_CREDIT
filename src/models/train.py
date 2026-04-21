"""
Training/Tuning Pipeline for Credit Risk Models.

Tuning and training models (ClassicMLP, EmbedMLP, XGBoost, RF)
on GermanCredit, GMSC, LendingClub datasets.
Ghi lại kết quả so sánh để tìm best model.

Usage:
    python -m src.models.train --dataset german_credit --n-trials 20
    python -m src.models.train --dataset gmsc --n-trials 20 --verbose
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score

from src.models.classic_mlp import get_classic_mlp_search_space, build_classic_mlp
from src.models.embed_mlp import get_embed_mlp_search_space, build_embed_mlp
from src.models.tree_models import build_random_forest, build_xgboost, get_random_forest_search_space, get_xgboost_search_space
from src.models.tunner import tune_sklearn_like_binary_model,tune_torch_binary_model
from src.models.loader import create_dataloaders, create_embedding_dataloaders
from src.models.trainer import predict
from src.data_processing.preprocess import CreditPreprocessor
from src.data_processing.domain_preprocess import GermanCreditDomainPreprocessor
from src.data_processing.utils import load_data, split_data, get_target_col

# ====================
# Configuration
# ====================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "src" / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def _resolve_dataset_name(dataset: str) -> str:
    if dataset == "germancredit":
        return "german"
    if dataset == "lendingclub":
        return "lending_club"
    return dataset


# ====================
# Data Loading
# ====================

def load_german_credit() -> tuple[pd.DataFrame, np.ndarray]:
    """Load GermanCredit dataset from UCI ML Repository (id=144)."""
    print("Fetching GermanCredit dataset from UCI ML Repository...")
    german_credit = fetch_ucirepo(id=144)
    
    X = german_credit.data.features.copy()
    y = german_credit.data.targets.copy()
    
    # Flatten y if needed
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].values
    else:
        y = np.asarray(y).reshape(-1)
    
    # Convert to binary (0/1)
    y = (y != 1).astype(int)
    
    print(f"Loaded GermanCredit: X={X.shape}, y={y.shape}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y

def load_gmsc() -> tuple[pd.DataFrame, np.ndarray]:
    """Load dữ liệu GMSC đã làm sạch từ file gmsc.csv."""
    file_path = DATA_DIR / "gmsc.csv"
    
    if not file_path.exists():
        # Dự phòng nếu bạn chưa di chuyển file vào thư mục data
        raise FileNotFoundError(f"Không tìm thấy file đã xử lý tại {file_path}")

    df = pd.read_csv(file_path)
    
    # Tách target (đảm bảo tên cột khớp với lúc bạn xuất file trong Notebook)
    target = "SeriousDlqin2yrs" 
    y = df[target].to_numpy().astype(int)
    X = df.drop(columns=[target]).copy()
    
    print(f"--- [DATA LOADED] GMSC (Full Cleaned): {X.shape} ---")
    return X, y
    
def load_lending_club() -> tuple[pd.DataFrame, np.ndarray]:
    """Load dữ liệu LendingClub đã xử lý từ file local."""
    file_path = DATA_DIR / "lendingclub.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Đảm bảo target được tách đúng
    y = df['target'].to_numpy().astype(int)
    X = df.drop(columns=['target']).copy()
    
    print(f"Loaded LendingClub Local: X={X.shape}, y={y.shape}")
    return X, y

def load_data(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load dataset by name."""
    loaders = {
        "germancredit": load_german_credit,
        "gmsc": load_gmsc,
        "lendingclub": load_lending_club,
    }
    
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(loaders.keys())}")
    
    return loaders[dataset]()


# ====================
# Preprocessing & Data Splitting
# ====================

def preprocess_data(
    df: pd.DataFrame,
    y: np.ndarray,
    dataset_name: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Leakage-safe preprocessing: split first, fit on train only, transform val/test."""
    print("\n" + "=" * 60)
    print("PREPROCESSING (LEAKAGE-SAFE)")
    print("=" * 60)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    y = np.asarray(y).reshape(-1).astype(int)

    # split raw features first
    X_trainval_df, X_test_df, y_trainval, y_test = train_test_split(
        df,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    val_ratio_in_trainval = val_size / (1.0 - test_size)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_trainval_df,
        y_trainval,
        test_size=val_ratio_in_trainval,
        random_state=seed,
        stratify=y_trainval,
    )

    dataset_name = _resolve_dataset_name(dataset_name)

    # model-specific preprocessors
    pp_classic = CreditPreprocessor(dataset_name=dataset_name, model_type="classic_mlp")
    pp_embedding = CreditPreprocessor(dataset_name=dataset_name, model_type="embedding")
    pp_tree = CreditPreprocessor(dataset_name=dataset_name, model_type="tree")

    pp_classic.fit(X_train_df)
    pp_embedding.fit(X_train_df)
    pp_tree.fit(X_train_df)

    X_train_classic = pp_classic.transform(X_train_df)
    X_val_classic = pp_classic.transform(X_val_df)
    X_test_classic = pp_classic.transform(X_test_df)

    X_train_embedding = pp_embedding.transform(X_train_df)
    X_val_embedding = pp_embedding.transform(X_val_df)
    X_test_embedding = pp_embedding.transform(X_test_df)

    X_train_tree = pp_tree.transform(X_train_df)
    X_val_tree = pp_tree.transform(X_val_df)
    X_test_tree = pp_tree.transform(X_test_df)

    emb_meta = pp_embedding.get_metadata()

    print(
        f"Train/Val/Test raw: {X_train_df.shape} / {X_val_df.shape} / {X_test_df.shape}"
    )
    print(
        f"Features classic/embedding/tree: {X_train_classic.shape[1]} / {X_train_embedding.shape[1]} / {X_train_tree.shape[1]}"
    )

    return {
        "preprocessors": {
            "classic": pp_classic,
            "embedding": pp_embedding,
            "tree": pp_tree,
        },
        "metadata": {
            "embedding": emb_meta,
        },
        "splits": {
            "y_train": np.asarray(y_train).astype(int),
            "y_val": np.asarray(y_val).astype(int),
            "y_test": np.asarray(y_test).astype(int),
            "classic": {
                "X_train": X_train_classic,
                "X_val": X_val_classic,
                "X_test": X_test_classic,
            },
            "embedding": {
                "X_train": X_train_embedding,
                "X_val": X_val_embedding,
                "X_test": X_test_embedding,
            },
            "tree": {
                "X_train": X_train_tree,
                "X_val": X_val_tree,
                "X_test": X_test_tree,
            },
        },
    }


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders."""
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def _split_num_cat(X_all: np.ndarray, cat_idxs: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not cat_idxs:
        return X_all.astype(np.float32), np.zeros((X_all.shape[0], 0), dtype=np.int64)
    x_cat = X_all[:, cat_idxs].astype(np.int64)
    num_idxs = [i for i in range(X_all.shape[1]) if i not in cat_idxs]
    x_num = X_all[:, num_idxs].astype(np.float32)
    return x_num, x_cat


def create_embedding_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cat_idxs: list[int],
    batch_size: int = 64,
) -> Dict[str, DataLoader]:
    xnum_tr, xcat_tr = _split_num_cat(X_train, cat_idxs)
    xnum_va, xcat_va = _split_num_cat(X_val, cat_idxs)
    xnum_te, xcat_te = _split_num_cat(X_test, cat_idxs)

    train_ds = TensorDataset(
        torch.from_numpy(xnum_tr).float(),
        torch.from_numpy(xcat_tr).long(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(xnum_va).float(),
        torch.from_numpy(xcat_va).long(),
        torch.from_numpy(y_val).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(xnum_te).float(),
        torch.from_numpy(xcat_te).long(),
        torch.from_numpy(y_test).long(),
    )

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }


# ====================
# Search Spaces
# ====================

def get_classic_mlp_search_space(input_dim: int) -> Dict[str, Any]:
    return {
        "input_dim": input_dim,
        "hidden_h1": [64, 128, 256],
        "hidden_h2": [32, 64, 128],
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
        "max_epochs": 100,
        "patience": 15,
    }

def get_embed_mlp_search_space(input_num_dim: int, cat_dims: list[int]) -> Dict[str, Any]:
    return {
        "input_num_dim": input_num_dim,
        "cat_dims": np.asarray(cat_dims, dtype=int),
        "emb_dims": None,
        "hidden_h1": [256, 512],
        "hidden_h2": [128, 256],
        "dropout": {"type": "float", "low": 0.3, "high": 0.5},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-5, "high": 1e-3},
        "max_epochs": 100,
        "patience": 15,
    }

def get_xgboost_search_space() -> Dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 0.5, "high": 2.0},
        "min_child_weight": {"type": "float", "low": 1.0, "high": 5.0},
        "random_state": 42,
        "n_jobs": -1,
    }


def get_random_forest_search_space() -> Dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
        "max_depth": [None, 10, 15, 20, 30],
        "min_samples_split": {"type": "int", "low": 2, "high": 10, "step": 2},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 5, "step": 1},
        "max_features": ["sqrt", "log2"],
        "random_state": 42,
        "n_jobs": -1,
    }


# ====================
# Model Builders
# ====================

def build_classic_mlp(cfg: Dict[str, Any]) -> ClassicMLP:
    cfg = dict(cfg)  # copy
    input_dim = cfg.pop("input_dim", None)
    if input_dim is None:
        raise ValueError("input_dim required")

    h1 = int(cfg.get("hidden_h1", 128))
    h2 = int(cfg.get("hidden_h2", 64))
    hidden_dims = (h1, h2)

    return ClassicMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=float(cfg.get("dropout", 0.3)),
    )


def build_embed_mlp(cfg: Dict[str, Any]) -> EmbedMLP:
    cfg = dict(cfg)
    input_num_dim = cfg.pop("input_num_dim", None)
    cat_dims = cfg.pop("cat_dims", [])
    if input_num_dim is None:
        raise ValueError("input_num_dim required")

    h1 = int(cfg.get("hidden_h1", 128))
    h2 = int(cfg.get("hidden_h2", 64))
    hidden_dims = (h1, h2)

    return EmbedMLP(
        input_num_dim=input_num_dim,
        cat_dims=cat_dims,
        emb_dims=cfg.get("emb_dims", None),
        hidden_dims=hidden_dims,
        dropout=float(cfg.get("dropout", 0.3)),
    )


def build_xgboost(cfg: Dict[str, Any]) -> Any:
    return get_xgboost_model(
        n_estimators=int(cfg.get("n_estimators", 300)),
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("max_depth", 4)),
        subsample=float(cfg.get("subsample", 0.9)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.9)),
        reg_alpha=float(cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(cfg.get("reg_lambda", 1.0)),
        min_child_weight=float(cfg.get("min_child_weight", 1.0)),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


def build_random_forest(cfg: Dict[str, Any]) -> Any:
    return get_random_forest_model(
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=cfg.get("max_depth", None),
        min_samples_split=int(cfg.get("min_samples_split", 2)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 1)),
        max_features=cfg.get("max_features", "sqrt"),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


# ====================
# Tuning
# ====================

def tune_all_models(
    data: Dict[str, Any],
    models: List[str],
    n_trials: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Tuning model với Optuna.
    Args:
        model: "embed_mlp", "classic_mlp", "xgboost", hoặc "random_forest"
        X_train, y_train: dữ liệu huấn luyện đã được preprocess (numpy arrays)
        X_valid, y_valid: dữ liệu validation đã được preprocess (numpy arrays)
        metadata: dict chứa thông tin về cat_idxs, cat_dims, actionable, immutable, etc.
        n_trials: số lượng trials cho tuning
        verbose: nếu True, in thêm thông tin chi tiết trong quá trình tuning

    Returns:
         Dictionary chứa:
         - "best_model": model đã được train với best hyperparameters
         - "best_score": PR-AUC trên validation của best_model
         - "best_params": hyperparameters tốt nhất tìm được
    """
    if model == "random_forest":
        search_space = get_random_forest_search_space()
        return tune_sklearn_like_binary_model(
            model_builder=build_random_forest,
            search_space=search_space,
            X_train=X_train,
            y_train=y_train,
            X_val=X_valid,
            y_val=y_valid,
            n_trials=n_trials,
            verbose=verbose,
            select_metric="pr_auc",
        )
    
    if model == "xgboost":
        search_space = get_xgboost_search_space()
        return tune_sklearn_like_binary_model(
            model_builder=build_xgboost,
            search_space=search_space,
            X_train=X_train,
            y_train=y_train,
            X_val=X_valid,
            y_val=y_valid,
            n_trials=n_trials,
            verbose=verbose,
            select_metric="pr_auc",
        )

    if model == "classic_mlp":
        train_loader = create_dataloaders(X=X_train, y=y_train, shuffle=True)
        valid_loader = create_dataloaders(X=X_valid, y=y_valid)
        search_space = get_classic_mlp_search_space(input_dim=X_train.shape[1])
        return tune_torch_binary_model(
            model_builder=build_classic_mlp,
            search_space=search_space,
            train_loader=train_loader,
            val_loader=valid_loader,
            n_trials=n_trials,
            verbose=verbose,
            select_metric="pr_auc",
        )
    
    if model == "embed_mlp":
        cat_idxs = metadata.get("cat_idxs", [])
        cat_dims = metadata.get("cat_dims", [])
        if not cat_dims:
            raise ValueError("embed_mlp requires metadata['cat_dims']")
        
        train_loader = create_embedding_dataloaders(X=X_train, y=y_train, cat_idxs=cat_idxs, shuffle=True)
        valid_loader = create_embedding_dataloaders(X=X_valid, y=y_valid, cat_idxs=cat_idxs)
        input_num_dim = X_train.shape[1] - len(cat_idxs)
        search_space = get_embed_mlp_search_space(input_num_dim=input_num_dim, cat_dims=cat_dims)
        return tune_torch_binary_model(
            model_builder=build_embed_mlp,
            search_space=search_space,
            train_loader=train_loader,
            val_loader=valid_loader,
            n_trials=n_trials,
            verbose=verbose,
            select_metric="auc",
        )

def evaluate_model(
    model: str,
    best_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Đánh giá model trên tập test với các metrics: accuracy, ROC-AUC, F1-score, PR-AUC.
    Args:
        model: "embed_mlp", "classic_mlp", "xgboost", hoặc "random_forest"
        best_model: model đã được train với best hyperparameters (từ tune_model)
        X_test, y_test: dữ liệu test đã được preprocess (numpy arrays)
        metadata: dict chứa thông tin về cat_idxs, cat_dims, actionable, immutable, etc.

    Returns:
        Dictionary chứa các metrics trên tập test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if model in ["random_forest", "xgboost"]:
        prob_np = best_model.predict_proba(X_test)
        if prob_np.ndim == 2:
            prob_np = prob_np[:, 1]
        pred_np = (prob_np >= 0.5).astype(int)
    elif model == "classic_mlp":
        test_loader = create_dataloaders(X=X_test, y=y_test)
        probs, preds = predict(
            model=best_model,
            test_loader=test_loader,
            device = device,
        )
        prob_np = probs.cpu().numpy()
        pred_np = preds.cpu().numpy()

    elif model == "embed_mlp":  
        cat_idxs = metadata.get("cat_idxs", [])
        if not cat_idxs:
            raise ValueError("embed_mlp requires metadata['cat_idxs']")
        
        test_loader = create_embedding_dataloaders(X=X_test, y=y_test, cat_idxs=cat_idxs)
        probs, preds = predict(
            model=best_model,
            test_loader=test_loader,
            device=device,
        )
        prob_np = probs.cpu().numpy()
        pred_np = preds.cpu().numpy()
    else:
        raise ValueError(f"Unknown model type '{model}' for evaluation.")
    
    return {
        "test_accuracy": accuracy_score(y_test, pred_np),
        "test_auc": roc_auc_score(y_test, prob_np),
        "test_f1_good": f1_score(y_test, pred_np),
        "test_f1_bad": f1_score(1 - y_test, 1 - pred_np),
        "test_pr_auc_good": average_precision_score(y_test, prob_np),
        "test_pr_auc_bad": average_precision_score(1 - y_test, 1 - prob_np),
    }

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj

def save_results(
    dataset: str,
    model: str,
    tuning_result: Dict[str, Any],
    eval_result: Dict[str, Any],
) -> None:
    """
    Lưu kết quả tuning và evaluation vào file JSON.
    Args:
        dataset: tên dataset
        model: tên model
        tunning_result: kết quả từ tune_model (best_model, best_score, best_params)
        eval_result: kết quả từ evaluate_model (metrics trên test set)
    """
    ds_results_dir = RESULTS_DIR / dataset
    ds_models_dir = MODELS_DIR / dataset
    ds_results_dir.mkdir(exist_ok=True)
    ds_models_dir.mkdir(exist_ok=True)

    best_model = tuning_result["best_model"]
    model_path = ds_models_dir / f"{model}_best.pkl"
    if isinstance(best_model, torch.nn.Module):
        torch.save(best_model.state_dict(), model_path)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

    best_cfg_file = ds_results_dir / "best_configs.json"
    best_configs = {}
    if best_cfg_file.exists():
        with open(best_cfg_file, "r") as f:
            best_configs = json.load(f)

    best_config = dict(tuning_result.get("best_config", {}))
    if model == "embed_mlp":
        resolved = getattr(best_model, "emb_dims", None)
        if resolved is not None:
            best_config["emb_dims"] = [int(d) for d in resolved]

    best_configs[model] = {
        "best_score": float(tuning_result.get("best_score", float("nan"))),
        "best_threshold": tuning_result.get("best_threshold", None),
        "best_threshold_f1": tuning_result.get("best_threshold_f1", None),
        "best_config": best_config,
    }
    with open(best_cfg_file, "w") as f:
        json.dump(_to_jsonable(best_configs), f, indent=2)

    eval_file = ds_results_dir / "eval_results.json"
    eval_results = {}
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_results = json.load(f)

    eval_results[model] = {
        **eval_result,
        "val_auc": float(tuning_result.get("best_score", float("nan"))),
        "best_threshold": tuning_result.get("best_threshold", None),
        "best_threshold_f1": tuning_result.get("best_threshold_f1", None),
        "best_config": best_config,
    }
    with open(eval_file, "w") as f:
        json.dump(_to_jsonable(eval_results), f, indent=2)

def main(
   dataset: str="german_credit",
   model: str="embed_mlp",
   n_trials: int=20,
   verbose: bool=False,     
) -> None:
    """
    Tuning và training model trên dataset với model đã chọn
    Args:
        dataset: "german_credit", "gmsc", hoặc "lending_club"
        model: "embed_mlp", "classic_mlp", "xgboost", hoặc "random_forest"
        n_trials: số lượng trials cho tuning (Optuna)
        verbose: nếu True, in thêm thông tin chi tiết trong quá trình tuning
    """

    """Main pipeline (single model per run)."""
    print("=" * 60)
    print(f"CREDIT RISK MODEL TUNING - {dataset.upper()} - {model.upper()}")
    print("=" * 60)

    df=load_data(dataset)
    # in ra phân phối lớp để check imbalance
    print("\nClass distribution in original data:")
    print(df[get_target_col(dataset)].value_counts(normalize=True))

    target_col = get_target_col(dataset)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df=df, 
        output_col=target_col
    )
    model_type_map = {
        "embed_mlp": "embedding",
        "classic_mlp": "classic",
        "xgboost": "tree",
        "random_forest": "tree",
    }
    
    if dataset == "german_credit":
        print("\nApplying Domain-Specific Preprocessing for German Credit...")
        domain_prep = GermanCreditDomainPreprocessor()
        X_train = domain_prep.transform(X_train)
        X_valid = domain_prep.transform(X_valid)
        X_test  = domain_prep.transform(X_test)
        # Lưu ý: Khi làm luồng XAI, bạn sẽ cần khởi tạo lại domain_prep 
        # để gọi hàm inverse_transform sau này.
    elif dataset == "gmsc":
        # Khởi tạo Domain Prep cho gmsc sau này
        pass
    elif dataset == "lending_club":
        # Khởi tạo Domain Prep cho lending_club sau này
        pass


    preprocessor = CreditPreprocessor(dataset_name=dataset, model_type=model_type_map.get(model))
    preprocessor.fit(X_train=X_train)
    X_train_pp = preprocessor.transform(X_train)
    X_valid_pp = preprocessor.transform(X_valid)
    X_test_pp = preprocessor.transform(X_test)
    metadata = preprocessor.get_metadata()

    y_train_pp = y_train.to_numpy().astype(int)
    y_valid_pp = y_valid.to_numpy().astype(int)
    y_test_pp = y_test.to_numpy().astype(int)

    tuning_result = tune_model(
        model=model,
        X_train=X_train_pp,
        y_train=y_train_pp,
        X_valid=X_valid_pp,
        y_valid=y_valid_pp,
        metadata=metadata,
        n_trials=n_trials,
        verbose=verbose,
    )

    print("\n" + "=" * 60)
    print(f"FINISHED TUNING {model.upper()} ON {dataset.upper()}")
    print("=" * 60)
    print(f"Best PR-AUC: {tuning_result['best_score']:.6f}")

    eval_result = evaluate_model(
        model=model,
        best_model=tuning_result["best_model"],
        X_test=X_test_pp,
        y_test=y_test_pp,
        metadata=metadata,
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ON TEST SET - {model.upper()} ON {dataset.upper()}")
    print("=" * 60)
    for metric, value in eval_result.items():
        print(f"{metric}: {value:.6f}")

    

    save_results(
        dataset=dataset,
        model=model,
        tuning_result=tuning_result,
        eval_result=eval_result,
    )

    print("\n" + "=" * 60)
    print(f"RESULTS SAVED AT {RESULTS_DIR / dataset}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and tune credit risk models.")
    parser.add_argument("--dataset", type=str, default="german_credit", choices=["german_credit", "gmsc", "lending_club"], help="Dataset to use.")
    parser.add_argument("--model", type=str, default="embed_mlp", choices=["embed_mlp", "classic_mlp", "xgboost", "random_forest"], help="Model to train and tune.")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of tuning trials (Optuna).")
    parser.add_argument("--verbose", action="store_true", help="If set, print detailed tuning information.")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model=args.model,
        n_trials=args.n_trials,
        verbose=args.verbose,
    )