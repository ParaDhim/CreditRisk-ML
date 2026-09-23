import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

def load_data():
    try:
        app_train = pd.read_csv('data/application_train.csv')
        bureau = pd.read_csv('data/bureau.csv')
        bureau_bal = pd.read_csv('data/bureau_balance.csv')
        prev_app = pd.read_csv('data/previous_application.csv')
        installments = pd.read_csv('data/installments_payments.csv')
        return app_train, bureau, bureau_bal, prev_app, installments
    except FileNotFoundError as e:
        print(f"Data loading failed: {e}")
        print("Please ensure you have downloaded the dataset according to data/README.md.")
        sys.exit(1)

def eda_summary(app_train):
    print("=== EDA Summary ===")
    print(f"Row count: {len(app_train)}")
    target_pct = app_train['TARGET'].mean() * 100
    print(f"Target distribution (% default): {target_pct:.2f}%")
    missing = app_train.isnull().mean() * 100
    top_missing = missing.sort_values(ascending=False).head(10)
    print("Top 10 columns missingness (%):")
    print(top_missing)
    print('===================\n')

def baseline_model(app_train):
    print("Running baseline model on application_train alone...")
    y = app_train['TARGET']
    X = app_train.drop(columns=['TARGET', 'SK_ID_CURR'])
    
    # Minimal cleaning
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'random_state': 42}
        
        model = lgb.train(params, dtrain, valid_sets=[dval], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        preds = model.predict(X_val)
        aucs.append(roc_auc_score(y_val, preds))
        
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"Baseline AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
    return mean_auc, std_auc

def feature_engineering(app_train, bureau, bureau_bal, prev_app, installments):
    print("Engineering features...")
    df = app_train.copy()
    
    # 1. Bureau & Bureau Balance
    bureau_bal_agg = bureau_bal.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].agg(['count', 'min', 'max']).reset_index()
    # rename columns
    bureau_bal_agg.columns = ['SK_ID_BUREAU', 'BB_COUNT', 'BB_MIN', 'BB_MAX']
    bureau = bureau.merge(bureau_bal_agg, on='SK_ID_BUREAU', how='left')
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'DAYS_CREDIT': ['mean', 'min'],
        'CREDIT_DAY_OVERDUE': 'mean',
        'AMT_CREDIT_SUM': 'sum',
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'BB_COUNT': 'mean'
    })
    bureau_agg.columns = ['_'.join(c).strip('_').upper() for c in bureau_agg.columns]
    bureau_agg['BUREAU_DEBT_RATIO'] = bureau_agg['AMT_CREDIT_SUM_DEBT_SUM'] / (bureau_agg['AMT_CREDIT_SUM_SUM'] + 1)
    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    
    # 2. Previous Applications
    prev_agg = prev_app.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'AMT_APPLICATION': 'mean',
        'AMT_CREDIT': 'mean',
        'NAME_CONTRACT_STATUS': lambda x: (x == 'Approved').mean()
    })
    prev_agg.columns = ['PREV_COUNT', 'PREV_AMT_APP_MEAN', 'PREV_AMT_CREDIT_MEAN', 'PREV_APPROVED_RATE']
    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
    
    # 3. Installments
    installments['LATE_PAYMENT'] = (installments['DAYS_ENTRY_PAYMENT'] > installments['DAYS_INSTALMENT']).astype(int)
    installments['PAYMENT_SHORTFALL'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    inst_agg = installments.groupby('SK_ID_CURR').agg({
        'LATE_PAYMENT': 'mean',
        'PAYMENT_SHORTFALL': 'mean'
    })
    inst_agg.columns = ['INST_LATE_RATE', 'INST_SHORTFALL_MEAN']
    df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
    
    # 4. Domain ratios on app_train base
    df['DEBT_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['EMP_TO_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1, skipna=True)
    
    # Drop columns with >80% missingness (unless domain justified)
    missing_pct = df.isnull().mean()
    to_drop = missing_pct[missing_pct > 0.8].index.tolist()
    if 'TARGET' in to_drop:
        to_drop.remove('TARGET')
    df.drop(columns=to_drop, inplace=True)
    
    print(f"Final feature count: {df.shape[1] - 2}") # excludes TARGET and SK_ID_CURR
    return df

def compare_imbalance_strategies(df):
    print("Comparing imbalance strategies...")
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    
    # Preprocess
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'weighted': {'auc': [], 'precision': [], 'recall': [], 'f1': []},
               'smote': {'auc': [], 'precision': [], 'recall': [], 'f1': []}}
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Strategy A: Class-weighted
        dtrain_w = lgb.Dataset(X_train, label=y_train)
        dval_w = lgb.Dataset(X_val, label=y_val, reference=dtrain_w)
        params_w = {'objective': 'binary', 'scale_pos_weight': 10, 'metric': 'auc', 'verbosity': -1, 'random_state': 42}
        model_w = lgb.train(params_w, dtrain_w, valid_sets=[dval_w], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        
        preds_w = model_w.predict(X_val)
        pred_labels_w = (preds_w > 0.5).astype(int)
        
        metrics['weighted']['auc'].append(roc_auc_score(y_val, preds_w))
        metrics['weighted']['precision'].append(precision_score(y_val, pred_labels_w))
        metrics['weighted']['recall'].append(recall_score(y_val, pred_labels_w))
        metrics['weighted']['f1'].append(f1_score(y_val, pred_labels_w))
        
        # Strategy B: SMOTE
        # Handle categories for SMOTE (SMOTENC is better but slow, simple fillna for demo, or ordinal encode)
        X_train_num = X_train.select_dtypes(exclude=['category']).fillna(X_train.median(numeric_only=True))
        X_val_num = X_val.select_dtypes(exclude=['category']).fillna(X_train.median(numeric_only=True))
        
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_num, y_train)
        
        model_sm = lgb.LGBMClassifier(random_state=42, objective='binary')
        model_sm.fit(X_train_sm, y_train_sm, eval_set=[(X_val_num, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        
        preds_sm = model_sm.predict_proba(X_val_num)[:, 1]
        pred_labels_sm = (preds_sm > 0.5).astype(int)
        
        metrics['smote']['auc'].append(roc_auc_score(y_val, preds_sm))
        metrics['smote']['precision'].append(precision_score(y_val, pred_labels_sm))
        metrics['smote']['recall'].append(recall_score(y_val, pred_labels_sm))
        metrics['smote']['f1'].append(f1_score(y_val, pred_labels_sm))
        
    print("\nResults (5-fold CV):")
    for strat, ms in metrics.items():
        print(f"  {strat.capitalize()}:")
        print(f"    AUC:       {np.mean(ms['auc']):.4f}")
        print(f"    Precision: {np.mean(ms['precision']):.4f}")
        print(f"    Recall:    {np.mean(ms['recall']):.4f}")
        print(f"    F1:        {np.mean(ms['f1']):.4f}")
        
    if np.mean(metrics['weighted']['auc']) > np.mean(metrics['smote']['auc']):
        best_strat = 'weighted'
        final_model = model_w  # approximation
    else:
        best_strat = 'smote'
        final_model = model_sm

    print(f"\nWinning strategy: {best_strat.capitalize()} loss wins on AUC.")
    return best_strat, metrics, final_model, X_val, y_val

def business_framing_and_shap(model, X_val, y_val, best_strat):
    print("\nBusiness Framing Analysis...")
    # Using last fold X_val for demonstration
    if best_strat == 'weighted':
        preds = model.predict(X_val)
    else:
        # X_val was processed differently for smote
        X_val_num = X_val.select_dtypes(exclude=['category']).fillna(X_val.median(numeric_only=True))
        preds = model.predict_proba(X_val_num)[:, 1]
        
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("\nThreshold Table:")
    print(f"{'Threshold':<10} | {'Approval Rate (%)':<18} | {'Recall (Defaults Caught %)':<26} | {'Precision (%)':<15}")
    print("-" * 75)
    for t in thresholds:
        approved = preds <= t
        app_rate = approved.mean() * 100
        
        pred_default = (preds > t).astype(int)
        recall = recall_score(y_val, pred_default) * 100
        prec = precision_score(y_val, pred_default, zero_division=0) * 100
        
        print(f"{t:<10.2f} | {app_rate:<18.2f} | {recall:<26.2f} | {prec:<15.2f}")
    
    print("\nRunning SHAP analysis on last validation fold...")
    if best_strat == 'weighted':
        # lightgbm native model
        explainer = shap.TreeExplainer(model)
        X_sample = X_val.sample(min(1000, len(X_val)), random_state=42)
        # handle categories for shap
        for c in X_sample.select_dtypes(['category']).columns:
            X_sample[c] = X_sample[c].cat.codes
        shap_values = explainer.shap_values(X_sample)
        # for binary classification lgb gives list of shap_values or numpy array
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[1]).mean(0)
        else:
            vals = np.abs(shap_values).mean(0)
        feat_imp = pd.DataFrame({'feature': X_sample.columns, 'importance': vals})
    else:
        # sklearn lgbm wrapper
        explainer = shap.TreeExplainer(model)
        X_sample = X_val.select_dtypes(exclude=['category']).fillna(X_val.median(numeric_only=True)).sample(min(1000, len(X_val)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            vals = np.abs(shap_values[1]).mean(0)
        else:
            vals = np.abs(shap_values).mean(0)
        feat_imp = pd.DataFrame({'feature': X_sample.columns, 'importance': vals})
        
    top_features = feat_imp.sort_values(by='importance', ascending=False).head(10)
    print("\nTop 10 SHAP features:")
    for ext, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    app_train, bureau, bureau_bal, prev_app, installments = load_data()
    eda_summary(app_train)
    baseline_auc, baseline_std = baseline_model(app_train)
    df = feature_engineering(app_train, bureau, bureau_bal, prev_app, installments)
    best_strat, metrics, final_model, X_val, y_val = compare_imbalance_strategies(df)
    business_framing_and_shap(final_model, X_val, y_val, best_strat)
