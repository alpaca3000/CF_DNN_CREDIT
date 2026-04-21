import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from .trainer import WeightedBCELoss, predict, set_prediction_threshold, train_model


# =========================
# Utilities
# =========================

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _suggest_from_space(trial: optuna.Trial, space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quy ước search space:
      - list/tuple: suggest_categorical
      - dict:
          {"type":"float","low":1e-4,"high":1e-2,"log":True}
          {"type":"int","low":32,"high":256,"step":32}
          {"type":"categorical","choices":[...]}
      - giá trị khác: giữ nguyên (constant)
    """
    cfg: Dict[str, Any] = {}
    for k, v in space.items():
        if isinstance(v, dict) and "type" in v:
            t = v["type"]
            if t == "float":
                low, high = float(v["low"]), float(v["high"])
                cfg[k] = trial.suggest_float(k, low, high, log=bool(v.get("log", False)))
            elif t == "int":
                low, high = int(v["low"]), int(v["high"])
                step = int(v.get("step", 1))
                cfg[k] = trial.suggest_int(k, low, high, step=step)
            elif t == "categorical":
                cfg[k] = trial.suggest_categorical(k, list(v["choices"]))
            else:
                raise ValueError(f"Unsupported space type: {t}")
        elif isinstance(v, (list, tuple)):
            cfg[k] = trial.suggest_categorical(k, list(v))
        else:
            cfg[k] = v
    return cfg


def _optuna_direction(select_metric: str) -> str:
    if select_metric == "logloss":
        return "minimize"
    return "maximize"


def _score_from_metrics(metrics: Dict[str, float], select_metric: str) -> float:
    if select_metric == "logloss":
        return metrics["logloss"] if not np.isnan(metrics["logloss"]) else float("inf")
    if select_metric == "neg_logloss":
        return -metrics["logloss"] if not np.isnan(metrics["logloss"]) else -float("inf")
    score = metrics.get(select_metric, float("nan"))
    if np.isnan(score):
        return -float("inf")
    return float(score)


def _merge_constant_params(best_params: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(best_params)
    for k, v in search_space.items():
        if isinstance(v, dict) and "type" in v:
            continue
        if isinstance(v, (list, tuple)):
            continue
        merged[k] = v
    return merged


def _collect_targets(loader: Iterable) -> np.ndarray:
    ys: List[np.ndarray] = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            if len(batch) in (2, 3):
                y = batch[-1]
            else:
                raise ValueError("Loader cần batch có y: (x,y) hoặc (x_num,x_cat,y).")
        else:
            raise ValueError("Batch phải là tuple/list có chứa y.")
        y_np = y.detach().cpu().numpy().reshape(-1)
        ys.append(y_np)
    if not ys:
        return np.array([], dtype=np.float32)
    return np.concatenate(ys, axis=0)


def _compute_class_weight_tensor(y_true: np.ndarray) -> torch.Tensor:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    classes = np.array([0, 1], dtype=int)
    if not np.isin(classes, np.unique(y_true)).all():
        raise ValueError("Train split phải chứa đủ cả 2 lớp 0 và 1 để tính class weight.")

    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_true)
    return torch.tensor(class_weights, dtype=torch.float32)


def _binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y_true, pred)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["auc"] = float("nan")
    try:
        out["logloss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    except Exception:
        out["logloss"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(1 - y_true, 1 - prob))
    except Exception:
        out["pr_auc"] = float("nan")
    return out


def _find_best_threshold_by_f1(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    prob = np.asarray(prob).reshape(-1).astype(float)

    if y_true.size == 0:
        raise ValueError("Không thể tune threshold trên tập rỗng.")

    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if thresholds.size == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1_scores = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])
    # y_true = np.asarray(y_true).reshape(-1).astype(int)
    # prob_good = np.asarray(prob).reshape(-1).astype(float)

    # if y_true.size == 0:
    #     raise ValueError("Không thể tune threshold trên tập rỗng.")

    # # 1. Đảo ngược dữ liệu để tập trung vào Lớp 0 (Nợ xấu)
    # # Vì 1 là Tốt, 0 là Xấu -> Lấy (1 - y) sẽ biến Xấu thành 1, Tốt thành 0
    # y_bad = 1 - y_true
    
    # # Xác suất là Nợ xấu = 1 - Xác suất Tốt
    # prob_bad = 1.0 - prob_good

    # # 2. Tính PR Curve cho Nợ xấu
    # precision, recall, thresholds_bad = precision_recall_curve(y_bad, prob_bad)
    
    # if thresholds_bad.size == 0:
    #     return 0.5, 0.0

    # precision = precision[:-1]
    # recall = recall[:-1]
    # denom = precision + recall
    # f1_scores = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)

    # # 3. Tìm vị trí F1 cao nhất cho Nợ xấu
    # best_idx = int(np.argmax(f1_scores))
    # best_thresh_bad = float(thresholds_bad[best_idx])
    # best_f1_bad = float(f1_scores[best_idx])

    # # 4. ĐỔI NGƯỠNG TRỞ LẠI HỆ QUY CHIẾU P(TỐT)
    # # Quy tắc: Bị coi là Nợ xấu nếu P(Xấu) >= thresh_bad
    # # Tương đương: 1 - P(Tốt) >= thresh_bad  --> P(Tốt) <= 1 - thresh_bad
    # # Vậy ngưỡng cần vượt qua để được tính là Tốt chính là: 1 - thresh_bad
    # best_thresh_good = 1.0 - best_thresh_bad

    # # Trả về: [Ngưỡng duyệt vay P(Tốt)], [F1 Score của Nợ xấu tại ngưỡng đó]
    # return best_thresh_good, best_f1_bad


@dataclass
class TrialResult:
    trial: int
    config: Dict[str, Any]
    score: float
    metrics: Dict[str, float]


# =========================
# Tuning for PyTorch models (ClassicMLP / EmbedMLP ...)
# =========================

def tune_torch_binary_model(
    model_builder: Callable[[Dict[str, Any]], nn.Module],
    train_loader: Iterable,
    val_loader: Iterable,
    search_space: Dict[str, Any],
    n_trials: int = 20,
    max_epochs: int = 100,
    patience: int = 10,
    min_delta: float = 0.0,
    optimizer_builder: Optional[
        Callable[[nn.Module, Dict[str, Any]], torch.optim.Optimizer]
    ] = None,
    criterion_builder: Optional[Callable[[Dict[str, Any]], nn.Module]] = None,
    select_metric: str = "auc",  # "auc" / "acc" / "neg_logloss"
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Tối ưu hyperparameter bằng Optuna cho model PyTorch binary classification.
    """
    set_seed(seed)
    sampler = optuna.samplers.TPESampler(seed=seed)
    direction = _optuna_direction(select_metric)
    study = optuna.create_study(direction=direction, sampler=sampler)

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_targets = _collect_targets(train_loader)
    class_weight_tensor = _compute_class_weight_tensor(train_targets)

    if criterion_builder is None:
        criterion_builder = lambda cfg: WeightedBCELoss(class_weight_tensor)

    if optimizer_builder is None:
        def optimizer_builder(m: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
            lr = float(cfg.get("lr", 1e-3))
            wd = float(cfg.get("weight_decay", 0.0))
            return torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)

    trials: List[TrialResult] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -float("inf") if direction == "maximize" else float("inf")

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_state, best_score
        cfg = _suggest_from_space(trial, search_space)

        model = model_builder(cfg)
        optimizer = optimizer_builder(model, cfg)
        criterion = criterion_builder(cfg)

        model, _history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            max_epochs=int(cfg.get("max_epochs", max_epochs)),
            patience=int(cfg.get("patience", patience)),
            min_delta=float(cfg.get("min_delta", min_delta)),
            device=device,
            verbose=False,
        )

        probs_t, _ = predict(model, val_loader, device=device)
        y_true = _collect_targets(val_loader)
        probs = probs_t.numpy().reshape(-1)

        metrics = _binary_metrics(y_true, probs)
        score = _score_from_metrics(metrics, select_metric)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("score", float(score))

        trials.append(
            TrialResult(
                trial=trial.number,
                config=cfg,
                score=float(score),
                metrics=metrics,
            )
        )

        if verbose:
            print(f"[Torch Tune] Trial {trial.number:02d}/{n_trials} | score={score:.6f}")

        is_better = (score > best_score) if direction == "maximize" else (score < best_score)
        if is_better:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())

        return float(score)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_cfg: Optional[Dict[str, Any]] = _merge_constant_params(dict(study.best_trial.params), search_space)

    if best_cfg is None or best_state is None or len(study.trials) == 0:
        raise RuntimeError("Tuning thất bại: không có trial hợp lệ.")

    best_model = model_builder(best_cfg)
    best_model.load_state_dict(best_state)
    best_model.to(device)

    val_probs_t, _ = predict(best_model, val_loader, device=device, threshold=0.5)
    val_targets = _collect_targets(val_loader)
    best_threshold, best_threshold_f1 = _find_best_threshold_by_f1(val_targets, val_probs_t.cpu().numpy())
    set_prediction_threshold(best_model, best_threshold)

    return {
        "study": study,
        "best_model": best_model,
        "best_config": best_cfg,
        "best_score": float(study.best_value),
        "best_threshold": float(best_threshold),
        "best_threshold_f1": float(best_threshold_f1),
        "trials": trials,
    }


# =========================
# Tuning for sklearn-like models (RandomForest, XGBoost)
# =========================

def tune_sklearn_like_binary_model(
    model_builder: Callable[[Dict[str, Any]], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    search_space: Dict[str, Any],
    n_trials: int = 20,
    select_metric: str = "auc",  # "auc" / "acc" / "neg_logloss"
    fit_fn: Optional[Callable[[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]], None]] = None,
    predict_proba_fn: Optional[Callable[[Any, np.ndarray], np.ndarray]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Tối ưu hyperparameter bằng Optuna cho mô hình có API gần sklearn:
      - fit(X, y) hoặc fit có valid set (qua fit_fn custom)
      - predict_proba(X) trả xác suất lớp 1
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    direction = _optuna_direction(select_metric)
    study = optuna.create_study(direction=direction, sampler=sampler)

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    trials: List[TrialResult] = []
    best_model: Optional[Any] = None
    best_score = -float("inf") if direction == "maximize" else float("inf")

    y_train = np.asarray(y_train).reshape(-1)
    y_val = np.asarray(y_val).reshape(-1)

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_model, best_score
        cfg = _suggest_from_space(trial, search_space)
        model = model_builder(cfg)

        if fit_fn is not None:
            fit_fn(model, X_train, y_train, X_val, y_val, cfg)
        else:
            model.fit(X_train, y_train)

        if predict_proba_fn is not None:
            prob = predict_proba_fn(model, X_val)
        else:
            prob = model.predict_proba(X_val)

        prob_1 = prob[:, 1] if prob.ndim == 2 else prob.reshape(-1)
        metrics = _binary_metrics(y_val, prob_1)
        score = _score_from_metrics(metrics, select_metric)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("config", cfg)
        trial.set_user_attr("score", float(score))

        trials.append(
            TrialResult(
                trial=trial.number,
                config=cfg,
                score=float(score),
                metrics=metrics,
            )
        )

        if verbose:
            print(f"[SK Tune] Trial {trial.number:02d}/{n_trials} | score={score:.6f}")

        is_better = (score > best_score) if direction == "maximize" else (score < best_score)
        if is_better:
            best_score = float(score)
            best_model = model

        return float(score)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_cfg: Optional[Dict[str, Any]] = _merge_constant_params(dict(study.best_trial.params), search_space)

    if best_cfg is None or best_model is None or len(study.trials) == 0:
        raise RuntimeError("Tuning thất bại: không có trial hợp lệ.")

    return {
        "study": study,
        "best_model": best_model,
        "best_config": best_cfg,
        "best_score": float(study.best_value),
        "trials": trials,
    }