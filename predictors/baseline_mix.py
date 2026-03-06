import os
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import joblib
import argparse
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from pathlib import Path
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR
def save_best_fold_model(model_dir, fold_metrics, fold_paths, metric_name='ROC_AUC'):
    scores = [m[metric_name] for m in fold_metrics]
    best_fold = int(np.argmax(scores))
    best_score = scores[best_fold]
    src = fold_paths[best_fold]
    dst = os.path.join(model_dir, f"best_model_fold_{best_fold}.joblib")
    shutil.copy(src, dst)
    print(f"   best fold: {best_fold} ({metric_name}={best_score:.4f}) → saved as best_model_fold_{best_fold}.joblib")


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall_Sensitivity': rec,
        'Specificity': specificity,
        'F1_score': f1,
        'ROC_AUC': roc_auc
    }

N_SPLITS = 5
RANDOM_STATE = 42

def train_default_models(X, y, base_dir):
    from copy import deepcopy
    results = {}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # --- XGBoost ---
    print("training XGBoost (default arguments)...")
    metrics_list = []
    model_dir = os.path.join(base_dir, "xgboost_default")
    os.makedirs(model_dir, exist_ok=True)
    fold_paths = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics_list.append(metrics)
        model_path = os.path.join(model_dir, f"fold_{fold}.joblib")
        joblib.dump(model, model_path)
        fold_paths.append(model_path)
    save_best_fold_model(model_dir, metrics_list, fold_paths, 'ROC_AUC')
    results['XGBoost (Default)'] = metrics_list

    # --- LightGBM ---
    print("training LightGBM (default arguments)...")
    metrics_list = []
    model_dir = os.path.join(base_dir, "lgbm_default")
    os.makedirs(model_dir, exist_ok=True)
    fold_paths = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics_list.append(metrics)
        model_path = os.path.join(model_dir, f"fold_{fold}.joblib")
        joblib.dump(model, model_path)
        fold_paths.append(model_path)
    save_best_fold_model(model_dir, metrics_list, fold_paths, 'ROC_AUC')
    results['LightGBM (Default)'] = metrics_list

    # --- CatBoost ---
    print("training CatBoost (default arguments)...")
    metrics_list = []
    model_dir = os.path.join(base_dir, "catboost_default")
    os.makedirs(model_dir, exist_ok=True)
    fold_paths = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = CatBoostClassifier(random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics_list.append(metrics)
        model_path = os.path.join(model_dir, f"fold_{fold}.joblib")
        joblib.dump(model, model_path)
        fold_paths.append(model_path)
    save_best_fold_model(model_dir, metrics_list, fold_paths, 'ROC_AUC')
    results['CatBoost (Default)'] = metrics_list

    # --- other model ---
    models_config = {
        'Random Forest': (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), False),
        'KNN': (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), True),
        'SVM': (SVC(probability=True, random_state=RANDOM_STATE), True),
        'SGD': (SGDClassifier(loss='log', random_state=RANDOM_STATE, max_iter=1000, tol=1e-3), True)
    }

    for name, (model_template, needs_scaler) in models_config.items():
        print(f"training {name} (default arguments)...")
        metrics_list = []
        model_dir = os.path.join(base_dir, f"{name.lower().replace(' ', '_')}_default")
        os.makedirs(model_dir, exist_ok=True)
        fold_paths = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if needs_scaler:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                model = deepcopy(model_template)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                y_proba = model.predict_proba(X_val_scaled)[:, 1]
                model_path = os.path.join(model_dir, f"fold_{fold}.joblib")
                joblib.dump((model, scaler), model_path)
            else:
                model = deepcopy(model_template)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
                model_path = os.path.join(model_dir, f"fold_{fold}.joblib")
                joblib.dump(model, model_path)

            metrics = compute_metrics(y_val, y_pred, y_proba)
            metrics_list.append(metrics)
            fold_paths.append(model_path)

        save_best_fold_model(model_dir, metrics_list, fold_paths, 'ROC_AUC')
        results[name + ' (Default)'] = metrics_list

    return results

def objective_xgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    return np.mean(scores)

def objective_lgb(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": RANDOM_STATE,
        "verbose": -1
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    return np.mean(scores)

def objective_cb(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 50, 300),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_strength": trial.suggest_float("random_strength", 0, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_state": RANDOM_STATE,
        "verbose": False,
        "allow_writing_files": False
    }
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    return np.mean(scores)

def tune_and_train_gbm(model_name, objective_func, X, y, base_dir, n_trials=20):
    print(f"Parameter tuning {model_name} ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_func(trial, X, y), n_trials=n_trials)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics_list = []
    model_dir = os.path.join(base_dir, f"{model_name.lower()}_tuned")
    os.makedirs(model_dir, exist_ok=True)
    fold_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_name == "XGBoost":
            model = XGBClassifier(**study.best_params, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
        elif model_name == "LightGBM":
            model = lgb.LGBMClassifier(**study.best_params, random_state=RANDOM_STATE, verbose=-1)
        elif model_name == "CatBoost":
            model = CatBoostClassifier(**study.best_params, random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_proba)
        metrics_list.append(metrics)
        model_path = os.path.join(model_dir, f"fold_{fold}.joblib")
        joblib.dump(model, model_path)
        fold_paths.append(model_path)

    save_best_fold_model(model_dir, metrics_list, fold_paths, 'ROC_AUC')
    return metrics_list

def main():
    dataset_name = args.name
    X1 = joblib.load(os.path.join(PROJECT_ROOT,"FPs", dataset_name,"data0.joblib"))
    X2 = joblib.load(os.path.join(PROJECT_ROOT,"FPs", dataset_name,"readout0.joblib"))
    X_1024 = X1[:, :1024]
    X = X1
    y = joblib.load(os.path.join(PROJECT_ROOT,"FPs", dataset_name,"label0.joblib"))
    y = y.ravel()
    BASE_DIR =  os.path.join(PROJECT_ROOT,"models_final_"+"mix_"+dataset_name)
    os.makedirs(BASE_DIR, exist_ok=True)

    default_results = train_default_models(X, y, BASE_DIR)
    tuned_results = {}
    for name, obj in [("XGBoost", objective_xgb), ("LightGBM", objective_lgb), ("CatBoost", objective_cb)]:
        metrics = tune_and_train_gbm(name, obj, X, y, BASE_DIR, n_trials=20)
        tuned_results[name + " (Tuned)"] = metrics
    all_results = {**default_results, **tuned_results}
    metric_names = ['Accuracy', 'Precision', 'Recall_Sensitivity', 'Specificity', 'F1_score', 'ROC_AUC']
    summary_rows = []
    print("\n" + "="*90)
    print("Model Performance Comparison (5-Fold CV, Mean ± SD)")
    print("="*90)

    for model_name, folds in all_results.items():
        row = {"Model": model_name}
        for metric in metric_names:
            values = [f[metric] for f in folds]
            mean_val = np.mean(values)
            std_val = np.std(values)
            row[f"{metric}_Mean"] = mean_val
            row[f"{metric}_Std"] = std_val
            if metric == "ROC_AUC":
                print(f"{model_name:25s}: {mean_val:.4f} ± {std_val:.4f}")
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"), index=False)
    print(f"\nAll models and results have been saved to '{BASE_DIR}'")
    print(f"Detailed Results: {BASE_DIR}/model_comparison.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', type=str, default="test", help='dataset name ')
    args = parser.parse_args()
    main()