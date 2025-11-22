"""
ML Model Training - Train direction, volatility, and regime models
"""

import pandas as pd
import json
import joblib
import os
import sys
from pathlib import Path
import numpy as np

# Fix encoding on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Installing...")
    os.system("pip install lightgbm")
    import lightgbm as lgb

from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

ART = Path("ml/artifacts")
ART.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ML MODEL TRAINING - Direction, Volatility, Regime")
print("="*70)

# Load dataset
print("\nLoading dataset...")
df = pd.read_parquet(ART / "dataset.parquet")
with open(ART / "feature_cols.json") as f:
    FEATURES = json.load(f)

print(f"Dataset loaded: {len(df)} samples, {len(FEATURES)} features")

# Time-based split (no shuffle!)
# Sort by date column if exists, otherwise by index
if 'date' in df.columns:
    df = df.sort_values(['date']).reset_index(drop=True)
else:
    df = df.reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

print(f"Train samples: {len(train)} ({len(train)/len(df)*100:.1f}%)")
print(f"Test samples:  {len(test)} ({len(test)/len(df)*100:.1f}%)")

# Prepare features and targets
X_train = train[FEATURES]
X_test = test[FEATURES]

y_train_dir = train['label_dir']
y_test_dir = test['label_dir']

y_train_vol = train['future_vol5']
y_test_vol = test['future_vol5']

y_train_regime = train['label_regime']
y_test_regime = test['label_regime']

# ========== MODEL 1: DIRECTION CLASSIFIER ==========
print("\n" + "="*70)
print("Training Direction Classifier (Up/Neutral/Down)")
print("="*70)

params_dir = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

dtrain_dir = lgb.Dataset(X_train, label=y_train_dir)
dval_dir = lgb.Dataset(X_test, label=y_test_dir, reference=dtrain_dir)

print("\nTraining...")
clf_dir = lgb.train(
    params_dir,
    dtrain_dir,
    num_boost_round=500,
    valid_sets=[dval_dir],
    valid_names=['test'],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
)

clf_dir.save_model(str(MODEL_DIR / "dir_model.txt"))
print(f"\nDirection model saved to {MODEL_DIR / 'dir_model.txt'}")

# Evaluate
yhat_dir = clf_dir.predict(X_test).argmax(axis=1)
acc_dir = accuracy_score(y_test_dir, yhat_dir)
print(f"\nTest Accuracy: {acc_dir:.4f}")

print("\nClassification Report:")
target_names = ['Down', 'Neutral', 'Up']
print(classification_report(y_test_dir, yhat_dir, target_names=target_names))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': clf_dir.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.0f}")

feature_importance.to_csv(ART / "feature_importance_dir.csv", index=False)

# ========== MODEL 2: VOLATILITY REGRESSOR ==========
print("\n" + "="*70)
print("Training Volatility Regressor (5-day vol)")
print("="*70)

params_vol = {
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'verbose': -1
}

dtrain_vol = lgb.Dataset(X_train, label=y_train_vol)
dval_vol = lgb.Dataset(X_test, label=y_test_vol, reference=dtrain_vol)

print("\nTraining...")
reg_vol = lgb.train(
    params_vol,
    dtrain_vol,
    num_boost_round=500,
    valid_sets=[dval_vol],
    valid_names=['test'],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
)

reg_vol.save_model(str(MODEL_DIR / "vol_model.txt"))
print(f"\nVolatility model saved to {MODEL_DIR / 'vol_model.txt'}")

# Evaluate
yhat_vol = reg_vol.predict(X_test)
mae_vol = mean_absolute_error(y_test_vol, yhat_vol)
rmse_vol = np.sqrt(((y_test_vol - yhat_vol) ** 2).mean())

print(f"\nTest MAE:  {mae_vol:.6f}")
print(f"Test RMSE: {rmse_vol:.6f}")

# ========== MODEL 3: REGIME CLASSIFIER ==========
print("\n" + "="*70)
print("Training Regime Classifier (Calm/Volatile/Crash)")
print("="*70)

params_regime = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'verbose': -1
}

dtrain_regime = lgb.Dataset(X_train, label=y_train_regime)
dval_regime = lgb.Dataset(X_test, label=y_test_regime, reference=dtrain_regime)

print("\nTraining...")
clf_regime = lgb.train(
    params_regime,
    dtrain_regime,
    num_boost_round=500,
    valid_sets=[dval_regime],
    valid_names=['test'],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
)

clf_regime.save_model(str(MODEL_DIR / "regime_model.txt"))
print(f"\nRegime model saved to {MODEL_DIR / 'regime_model.txt'}")

# Evaluate
yhat_regime = clf_regime.predict(X_test).argmax(axis=1)
acc_regime = accuracy_score(y_test_regime, yhat_regime)
print(f"\nTest Accuracy: {acc_regime:.4f}")

print("\nClassification Report:")
regime_names = ['Calm', 'Volatile', 'Crash']
print(classification_report(y_test_regime, yhat_regime, target_names=regime_names))

# ========== SAVE TRAINING LOG ==========
print("\n" + "="*70)
print("Saving Training Logs")
print("="*70)

log_content = f"""ML Model Training Log
=====================
Date: {pd.Timestamp.now()}

Dataset:
  Total samples: {len(df)}
  Train samples: {len(train)}
  Test samples:  {len(test)}
  Features: {len(FEATURES)}

Model 1: Direction Classifier
  Objective: Multiclass (Up/Neutral/Down)
  Test Accuracy: {acc_dir:.4f}
  Classes: Down=0, Neutral=1, Up=2

Model 2: Volatility Regressor
  Objective: Regression (5-day volatility)
  Test MAE: {mae_vol:.6f}
  Test RMSE: {rmse_vol:.6f}

Model 3: Regime Classifier
  Objective: Multiclass (Calm/Volatile/Crash)
  Test Accuracy: {acc_regime:.4f}
  Classes: Calm=0, Volatile=1, Crash=2

Top 10 Features (by importance):
"""

for i, row in feature_importance.head(10).iterrows():
    log_content += f"  {i+1}. {row['feature']:20s}: {row['importance']:.0f}\n"

log_content += f"\nModels saved to: ml/models/\n"
log_content += f"Artifacts saved to: ml/artifacts/\n"

with open(ART / "train_log.txt", "w") as f:
    f.write(log_content)

print(f"\nTraining log saved to {ART / 'train_log.txt'}")

# Save model metadata
metadata = {
    'training_date': str(pd.Timestamp.now()),
    'dataset_size': len(df),
    'train_size': len(train),
    'test_size': len(test),
    'num_features': len(FEATURES),
    'models': {
        'direction': {
            'file': 'dir_model.txt',
            'accuracy': float(acc_dir),
            'classes': {'0': 'Down', '1': 'Neutral', '2': 'Up'}
        },
        'volatility': {
            'file': 'vol_model.txt',
            'mae': float(mae_vol),
            'rmse': float(rmse_vol)
        },
        'regime': {
            'file': 'regime_model.txt',
            'accuracy': float(acc_regime),
            'classes': {'0': 'Calm', '1': 'Volatile', '2': 'Crash'}
        }
    }
}

with open(ART / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nFiles created:")
print(f"  - {MODEL_DIR / 'dir_model.txt'}")
print(f"  - {MODEL_DIR / 'vol_model.txt'}")
print(f"  - {MODEL_DIR / 'regime_model.txt'}")
print(f"  - {ART / 'train_log.txt'}")
print(f"  - {ART / 'model_metadata.json'}")
print(f"  - {ART / 'feature_importance_dir.csv'}")

print("\nNext steps:")
print("  1. Run: python ml/backtest.py")
print("  2. Start API: uvicorn ml.api:app --reload --port 8001")
