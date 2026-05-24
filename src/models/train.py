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
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

def tune_model(
    model: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    metadata: Dict[str, Any],
    n_trials: int,
    verbose: bool,
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
    ds_output_dir = OUTPUT_DIR / dataset
    ds_models_dir = ds_output_dir / "models"
    ds_output_dir.mkdir(exist_ok=True)
    ds_models_dir.mkdir(exist_ok=True)

    best_model = tuning_result["best_model"]
    model_path = ds_models_dir / f"{model}_best.pkl"
    if isinstance(best_model, torch.nn.Module):
        torch.save(best_model.state_dict(), model_path)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

    best_cfg_file = ds_models_dir / "best_configs.json"
    best_configs = {}
    if best_cfg_file.exists():
        with open(best_cfg_file, "r") as f:
            best_configs = json.load(f)

    best_config = dict(tuning_result.get("best_config", {}))
    if model == "embed_mlp":
        resolved = getattr(best_model, "emb_dims", None)
        if resolved is not None:
            best_config["emb_dims"] = [int(d) for d in resolved]

    # Chỉ lưu cấu hình tốt nhất cho từng model
    best_configs[model] = best_config
    with open(best_cfg_file, "w") as f:
        json.dump(_to_jsonable(best_configs), f, indent=2)

    eval_file = ds_models_dir / "eval_results.json"
    eval_results = {}
    if eval_file.exists():
        with open(eval_file, "r") as f:
            eval_results = json.load(f)

    # Chỉ lưu metrics/kết quả đánh giá cho từng model
    eval_results[model] = {
        **eval_result,
        "val_pr_auc": float(tuning_result.get("best_score", float("nan"))),
        "best_threshold": tuning_result.get("best_threshold", None),
        "best_threshold_f1": tuning_result.get("best_threshold_f1", None),
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
        target_col=target_col
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
    elif dataset == "lendingclub":
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
    print(f"RESULTS SAVED AT {OUTPUT_DIR / dataset / 'models'}")
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
