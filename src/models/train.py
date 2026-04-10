"""
Training/Tuning Pipeline for Credit Risk Models.

Tuning and training models (ClassicMLP, EmbedMLP, XGBoost, RF)
on GermanCredit, GMSC, LendingClub datasets.
Ghi lại kết quả so sánh để tìm best model.

Usage:
    python -m src.models.train --dataset german_credit --n- trials 20
    python -m src.models.train --dataset gmsc --n-trials 20 --verbose
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.classic_mlp import ClassicMLP
from src.models.embed_mlp import EmbedMLP
from src.models.tree_models import get_random_forest_model, get_xgboost_model
from src.models.tunner import (
    tune_sklearn_like_binary_model,
    tune_torch_binary_model,
)
from src.models.trainer import predict
from src.preprocess.preprocess import CreditPreprocessor

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


# ====================
# Data Loading
# ====================

def load_german_credit() -> tuple[pd.DataFrame, np.ndarray]:
    """Load GermanCredit dataset from data/german_credit.csv."""
    print("Fetching GermanCredit dataset from data/german_credit.csv...")
    df = pd.read_csv(DATA_DIR / "german_credit.csv")
    
    # GermanCredit target is 'Class'
    target_col = "Class"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {df.columns.tolist()}")
    
    y = df[target_col].to_numpy().astype(int)
    X = df.drop(columns=[target_col]).copy()
    
    print(f"Loaded GermanCredit: X={X.shape}, y={y.shape}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y

# def load_german_credit() -> tuple[pd.DataFrame, np.ndarray]:
#     german = fetch_ucirepo(id=144)
#     X = german.data.features.copy()
#     y_raw = german.data.targets.copy()

#     if isinstance(y_raw, pd.DataFrame):
#         y_raw = y_raw.iloc[:, 0].to_numpy()
#     else:
#         y_raw = np.asarray(y_raw).reshape(-1)

#     print("Unique raw target values:", np.unique(y_raw, return_counts=True))

#     # UCI German Credit: 1=good, 2=bad
#     y = (y_raw == 1).astype(int)  # bad=1, good=0

#     print(f"Loaded GermanCredit from UCI: X={X.shape}, y={y.shape}")
#     print(f"  Class distribution: {np.bincount(y)}")
#     return X, y


def load_gmsc() -> tuple[pd.DataFrame, np.ndarray]:
    """Load GMSC dataset from local CSV files."""
    gmsc = pd.read_csv(DATA_DIR / "gmsc.csv")
    
    # GMSC target is 'SeriousDlqin2yrs'
    target_col = "SeriousDlqin2yrs"
    if target_col not in gmsc.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {gmsc.columns.tolist()}")
    
    y = gmsc[target_col].to_numpy().astype(int)
    X = gmsc.drop(columns=[target_col]).copy()
    
    print(f"Loaded GMSC: X={X.shape}, y={y.shape}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")
    
    return X, y


def load_lending_club() -> tuple[pd.DataFrame, np.ndarray]:
    """Load LendingClub dataset from local CSV files."""
    print("Fetching LendingClub dataset from  data/lending_club_balanced_sample_10k.csv...")
    lending_club = pd.read_csv(DATA_DIR / "lending_club_balanced_sample_10k.csv")

    target_col = "target"
    if target_col not in lending_club.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {lending_club.columns.tolist()}")
    
    y = lending_club[target_col].to_numpy().astype(int)
    X = lending_club.drop(columns=[target_col]).copy()

    print(f"Loaded LendingClub: X={X.shape}, y={y.shape}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")

    return X, y


def load_data(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load dataset by name."""
    loaders = {
        "german_credit": load_german_credit,
        "gmsc": load_gmsc,
        "lending_club": load_lending_club,
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
    model: str = "all",
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

    all_models = {"classic_mlp", "embed_mlp", "xgboost", "random_forest"}
    if model not in all_models and model != "all":
        raise ValueError(f"Unknown model: {model}. Choose from ['all', {sorted(all_models)}]")

    need_classic = model in {"all", "classic_mlp"}
    need_embedding = model in {"all", "embed_mlp"}
    need_tree = model in {"all", "xgboost", "random_forest"}

    preprocessors: Dict[str, CreditPreprocessor] = {}
    metadata: Dict[str, Any] = {}
    splits: Dict[str, Any] = {
        "y_train": np.asarray(y_train).astype(int),
        "y_val": np.asarray(y_val).astype(int),
        "y_test": np.asarray(y_test).astype(int),
    }

    if need_classic:
        pp_classic = CreditPreprocessor(dataset_name=dataset_name, model_type="classic_mlp")
        pp_classic.fit(X_train_df)
        preprocessors["classic"] = pp_classic
        splits["classic"] = {
            "X_train": pp_classic.transform(X_train_df),
            "X_val": pp_classic.transform(X_val_df),
            "X_test": pp_classic.transform(X_test_df),
        }

    if need_embedding:
        pp_embedding = CreditPreprocessor(dataset_name=dataset_name, model_type="embedding")
        pp_embedding.fit(X_train_df)
        preprocessors["embedding"] = pp_embedding
        splits["embedding"] = {
            "X_train": pp_embedding.transform(X_train_df),
            "X_val": pp_embedding.transform(X_val_df),
            "X_test": pp_embedding.transform(X_test_df),
        }
        metadata["embedding"] = pp_embedding.get_metadata()

    if need_tree:
        pp_tree = CreditPreprocessor(dataset_name=dataset_name, model_type="tree")
        pp_tree.fit(X_train_df)
        preprocessors["tree"] = pp_tree
        splits["tree"] = {
            "X_train": pp_tree.transform(X_train_df),
            "X_val": pp_tree.transform(X_val_df),
            "X_test": pp_tree.transform(X_test_df),
        }

    print(f"Train/Val/Test raw: {X_train_df.shape} / {X_val_df.shape} / {X_test_df.shape}")
    dim_msgs = []
    if "classic" in splits:
        dim_msgs.append(f"classic={splits['classic']['X_train'].shape[1]}")
    if "embedding" in splits:
        dim_msgs.append(f"embedding={splits['embedding']['X_train'].shape[1]}")
    if "tree" in splits:
        dim_msgs.append(f"tree={splits['tree']['X_train'].shape[1]}")
    if dim_msgs:
        print("Features: " + " | ".join(dim_msgs))

    return {
        "preprocessors": preprocessors,
        "metadata": metadata,
        "splits": splits,
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
        # keep constant (not an Optuna categorical choice)
        "cat_dims": np.asarray(cat_dims, dtype=int),
        "emb_dims": None,
        "hidden_h1": [64, 128, 256],
        "hidden_h2": [32, 64, 128],
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
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
    """Tune all 5 models."""
    print("\n" + "=" * 60)
    print("TUNING MODELS")
    print("=" * 60)
    
    results = {}
    y_train = data["splits"]["y_train"]
    y_val = data["splits"]["y_val"]
    y_test = data["splits"]["y_test"]

    if "classic_mlp" in models:
        if "classic" not in data["splits"]:
            raise ValueError("Missing classic preprocessing splits for classic_mlp.")
        classic = data["splits"]["classic"]
        classic_loaders = create_dataloaders(
            classic["X_train"],
            y_train,
            classic["X_val"],
            y_val,
            classic["X_test"],
            y_test,
        )
        input_dim_classic = classic["X_train"].shape[1]

        print("\n[ClassicMLP] Tuning...")
        space = get_classic_mlp_search_space(input_dim_classic)
        result = tune_torch_binary_model(
            model_builder=build_classic_mlp,
            train_loader=classic_loaders["train_loader"],
            val_loader=classic_loaders["val_loader"],
            search_space=space,
            n_trials=n_trials,
            select_metric="pr_auc",
            verbose=verbose,
        )
        results["classic_mlp"] = result
        print(f"  Best PR-AUC: {result['best_score']:.6f}")
        print(f"  Best threshold: {result.get('best_threshold', 0.5):.6f}")
    
    if "embed_mlp" in models:
        if "embedding" not in data["splits"] or "embedding" not in data["metadata"]:
            raise ValueError("Missing embedding preprocessing splits/metadata for embed_mlp.")
        embedding = data["splits"]["embedding"]
        emb_meta = data["metadata"]["embedding"]
        embedding_loaders = create_embedding_dataloaders(
            embedding["X_train"],
            y_train,
            embedding["X_val"],
            y_val,
            embedding["X_test"],
            y_test,
            cat_idxs=emb_meta["cat_idxs"],
        )
        input_num_dim = embedding["X_train"].shape[1] - len(emb_meta["cat_idxs"])

        print("\n[EmbedMLP] Tuning...")
        space = get_embed_mlp_search_space(input_num_dim=input_num_dim, cat_dims=emb_meta["cat_dims"])
        result = tune_torch_binary_model(
            model_builder=build_embed_mlp,
            train_loader=embedding_loaders["train_loader"],
            val_loader=embedding_loaders["val_loader"],
            search_space=space,
            n_trials=n_trials,
            select_metric="pr_auc",
            verbose=verbose,
        )
        results["embed_mlp"] = result
        print(f"  Best PR-AUC: {result['best_score']:.6f}")
        print(f"  Best threshold: {result.get('best_threshold', 0.5):.6f}")
    
    if "xgboost" in models:
        if "tree" not in data["splits"]:
            raise ValueError("Missing tree preprocessing splits for xgboost.")
        tree = data["splits"]["tree"]
        print("\n[XGBoost] Tuning...")
        space = get_xgboost_search_space()
        result = tune_sklearn_like_binary_model(
            model_builder=build_xgboost,
            X_train=tree["X_train"],
            y_train=y_train,
            X_val=tree["X_val"],
            y_val=y_val,
            search_space=space,
            n_trials=n_trials,
            select_metric="pr_auc",
            verbose=verbose,
        )
        results["xgboost"] = result
        print(f"  Best PR-AUC: {result['best_score']:.6f}")
    
    if "random_forest" in models:
        if "tree" not in data["splits"]:
            raise ValueError("Missing tree preprocessing splits for random_forest.")
        tree = data["splits"]["tree"]
        print("\n[RandomForest] Tuning...")
        space = get_random_forest_search_space()
        result = tune_sklearn_like_binary_model(
            model_builder=build_random_forest,
            X_train=tree["X_train"],
            y_train=y_train,
            X_val=tree["X_val"],
            y_val=y_val,
            search_space=space,
            n_trials=n_trials,
            select_metric="pr_auc",
            verbose=verbose,
        )
        results["random_forest"] = result
        print(f"  Best PR-AUC: {result['best_score']:.6f}")
    
    return results


# ====================
# Evaluation
# ====================

def evaluate_models(
    tuning_results: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate all best models on test set."""
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    y_train = data["splits"]["y_train"]
    y_val = data["splits"]["y_val"]
    y_test = data["splits"]["y_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classic_loaders = None
    embedding_loaders = None
    
    eval_results = {}
    
    for model_name, tuning_result in tuning_results.items():
        print(f"\nEvaluating {model_name}...")
        best_model = tuning_result["best_model"]
        best_config = tuning_result["best_config"]
        
        try:
            if model_name in ["classic_mlp", "embed_mlp"]:
                if model_name == "classic_mlp":
                    if classic_loaders is None:
                        classic = data["splits"]["classic"]
                        classic_loaders = create_dataloaders(
                            classic["X_train"],
                            y_train,
                            classic["X_val"],
                            y_val,
                            classic["X_test"],
                            y_test,
                        )
                else:
                    if embedding_loaders is None:
                        embedding = data["splits"]["embedding"]
                        emb_meta = data["metadata"]["embedding"]
                        embedding_loaders = create_embedding_dataloaders(
                            embedding["X_train"],
                            y_train,
                            embedding["X_val"],
                            y_val,
                            embedding["X_test"],
                            y_test,
                            cat_idxs=emb_meta["cat_idxs"],
                        )

                test_loader = (
                    classic_loaders["test_loader"]
                    if model_name == "classic_mlp"
                    else embedding_loaders["test_loader"]
                )
                probs, preds = predict(best_model, test_loader, device=device)
                prob_np = probs.cpu().numpy()
                pred_np = preds.cpu().numpy()
            else:
                X_test = data["splits"]["tree"]["X_test"]
                prob_np = best_model.predict_proba(X_test)
                if prob_np.ndim == 2:
                    prob_np = prob_np[:, 1]
                pred_np = (prob_np >= 0.5).astype(int)
            
            acc = accuracy_score(y_test, pred_np)
            auc = roc_auc_score(y_test, prob_np)
            f1 = f1_score(y_test, pred_np)
            pr_auc = average_precision_score(1 - y_test, 1 - prob_np)
            
            eval_results[model_name] = {
                "test_accuracy": float(acc),
                "test_auc": float(auc),
                "test_f1": float(f1),
                "test_pr_auc": float(pr_auc),
                "val_auc": float(tuning_result["best_score"]),
                "best_threshold": tuning_result.get("best_threshold", None),
                "best_threshold_f1": tuning_result.get("best_threshold_f1", None),
                "best_config": best_config,
            }
            
            print(f"  Accuracy: {acc:.6f}")
            print(f"  AUC: {auc:.6f}")
            print(f"  PR AUC: {pr_auc:.6f}")
            print(f"  F1: {f1:.6f}")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            eval_results[model_name] = {"error": str(e)}
    
    return eval_results


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


# ====================
# Save Results
# ====================

def save_results(
    dataset: str,
    tuning_results: Dict[str, Any],
    eval_results: Dict[str, Any],
) -> None:
    """Save results to JSON and models to pickle."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Create dataset-specific output dirs
    ds_results_dir = RESULTS_DIR / dataset
    ds_models_dir = MODELS_DIR / dataset
    ds_results_dir.mkdir(exist_ok=True)
    ds_models_dir.mkdir(exist_ok=True)
    
    # Summary
    summary = {}
    for model_name, eval_res in eval_results.items():
        if "error" not in eval_res:
            summary[model_name] = {
                "test_accuracy": eval_res["test_accuracy"],
                "test_auc": eval_res["test_auc"],
                "test_f1": eval_res["test_f1"],
                "test_pr_auc": eval_res["test_pr_auc"],
                "val_auc": eval_res["val_auc"],
                "best_threshold": eval_res.get("best_threshold", None),
                "best_threshold_f1": eval_res.get("best_threshold_f1", None),
            }
    
    summary_file = ds_results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")
    
    # Full results
    results_file = ds_results_dir / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(_to_jsonable(eval_results), f, indent=2)
    print(f"Saved eval results to {results_file}")

    # Best configs per model (for debugging/reproducibility)
    best_configs = {}
    for model_name, tuning_result in tuning_results.items():
        best_configs[model_name] = {
            "best_score": float(tuning_result.get("best_score", float("nan"))),
            "best_threshold": tuning_result.get("best_threshold", None),
            "best_threshold_f1": tuning_result.get("best_threshold_f1", None),
            "best_config": tuning_result.get("best_config", {}),
        }
    best_cfg_file = ds_results_dir / "best_configs.json"
    with open(best_cfg_file, "w") as f:
        json.dump(_to_jsonable(best_configs), f, indent=2)
    print(f"Saved best configs to {best_cfg_file}")
    
    # Best models
    for model_name, tuning_result in tuning_results.items():
        best_model = tuning_result["best_model"]
        model_path = ds_models_dir / f"{model_name}_best.pkl"
        
        try:
            if isinstance(best_model, torch.nn.Module):
                torch.save(best_model.state_dict(), model_path)
            else:
                with open(model_path, "wb") as f:
                    pickle.dump(best_model, f)
            print(f"Saved {model_name} to {model_path}")
        except Exception as e:
            print(f"Failed to save {model_name}: {e}")


# ====================
# Main
# ====================

def main(
    dataset: str = "german_credit",
    model: str = "all",
    n_trials: int = 10,
    verbose: bool = False,
) -> None:
    """Main pipeline."""
    print("=" * 60)
    print(f"CREDIT RISK MODEL TUNING - {dataset.upper()}")
    print("=" * 60)
    
    # Load
    df, y = load_data(dataset)
    
    # Preprocess
    data = preprocess_data(df, y, dataset_name=dataset, model=model)
    
    all_models = ["classic_mlp", "embed_mlp", "xgboost", "random_forest"]
    selected_models = all_models if model == "all" else [model]

    # Tune
    tuning_results = tune_all_models(data, models=selected_models, n_trials=n_trials, verbose=verbose)
    
    # Evaluate
    eval_results = evaluate_models(tuning_results, data)
    
    # Save
    save_results(dataset, tuning_results, eval_results)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit risk model tuning pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="german_credit",
        choices=["german_credit", "gmsc", "lending_club"],
        help="Dataset to use (currently only german_credit is supported)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "classic_mlp", "embed_mlp", "xgboost", "random_forest"],
        help="Tune/evaluate one model or all models",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of tuning trials per model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output from Optuna",
    )
    
    args = parser.parse_args()
    main(dataset=args.dataset, model=args.model, n_trials=args.n_trials, verbose=args.verbose)
