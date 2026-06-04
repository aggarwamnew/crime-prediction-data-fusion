"""
01_lstm_baseline.py — LSTM vs Random Forest comparison.

ISOLATED EXPERIMENT: Reads only from existing data. No writes to main project files.
Results saved to notebooks/experimental/results/ only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from src.data.db import ThesisDB
from src.utils.config import PROJECT_ROOT
import json, os

RESULTS_DIR = PROJECT_ROOT / "notebooks/experimental/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA (READ-ONLY)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LSTM vs RANDOM FOREST COMPARISON")
print("=" * 70)
print("\nLoading data (read-only)...")

db = ThesisDB()
crime = db.query("""
    SELECT lsoa_code, month, COUNT(*) as crime_count
    FROM crime_clean
    GROUP BY lsoa_code, month
""")

# Filter active LSOAs (same as main experiments)
lsoa_totals = crime.groupby('lsoa_code')['crime_count'].sum()
active = lsoa_totals[lsoa_totals >= 36].index
crime = crime[crime['lsoa_code'].isin(active)]

all_months = sorted(crime['month'].unique())
all_lsoas = sorted(crime['lsoa_code'].unique())
grid = pd.MultiIndex.from_product([all_lsoas, all_months], names=['lsoa_code', 'month'])
df = crime.set_index(['lsoa_code', 'month']).reindex(grid, fill_value=0).reset_index()
df = df.sort_values(['lsoa_code', 'month']).reset_index(drop=True)
db.close()

print(f"  LSOAs: {len(all_lsoas):,}")
print(f"  Months: {len(all_months)}")
print(f"  Total rows: {len(df):,}")

# ══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING (same as main baseline)
# ══════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

for lag in [1, 2, 3, 6, 12]:
    df[f'lag_{lag}'] = df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3, 6, 12]:
    df[f'rolling_mean_{w}'] = df.groupby('lsoa_code')['crime_count'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean())
df['month_sin'] = np.sin(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * pd.to_datetime(df['month']).dt.month / 12)
ts = pd.to_datetime(df['month'])
df['time_idx'] = (ts.dt.year - ts.dt.year.min()) * 12 + ts.dt.month

base_features = ['lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                 'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
                 'month_sin', 'month_cos', 'time_idx']

df_model = df.dropna()
test_months = all_months[-6:]
train = df_model[~df_model['month'].isin(test_months)]
test = df_model[df_model['month'].isin(test_months)]

print(f"  Features: {len(base_features)}")
print(f"  Train: {len(train):,}, Test: {len(test):,}")

# ══════════════════════════════════════════════════════════════════════════
# 3. RANDOM FOREST BASELINE (for comparison)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RANDOM FOREST BASELINE")
print("=" * 70)

rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5,
                           n_jobs=-1, random_state=42)
rf.fit(train[base_features], train['crime_count'])
rf_pred = rf.predict(test[base_features])
rf_r2 = r2_score(test['crime_count'], rf_pred)
rf_mae = mean_absolute_error(test['crime_count'], rf_pred)
print(f"  R2:  {rf_r2:.4f}")
print(f"  MAE: {rf_mae:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 4. LSTM MODEL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LSTM MODEL")
print("=" * 70)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train[base_features].values)
X_test = scaler_X.transform(test[base_features].values)
y_train = scaler_y.fit_transform(train['crime_count'].values.reshape(-1, 1)).ravel()
y_test_raw = test['crime_count'].values

# Reshape for LSTM: (samples, seq_len=1, features)
# Using single-step input (same features as RF for fair comparison)
X_train_t = torch.FloatTensor(X_train).unsqueeze(1)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test).unsqueeze(1)

# DataLoader
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"  Device: {device}")

model = LSTMModel(input_size=len(base_features), hidden_size=64, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# Training
EPOCHS = 50
print(f"  Training {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f"    Epoch {epoch+1:3d}/{EPOCHS}: loss={avg_loss:.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    lstm_pred_scaled = model(X_test_t.to(device)).cpu().numpy()

lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).ravel()
lstm_r2 = r2_score(y_test_raw, lstm_pred)
lstm_mae = mean_absolute_error(y_test_raw, lstm_pred)
print(f"\n  R2:  {lstm_r2:.4f}")
print(f"  MAE: {lstm_mae:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)
print(f"  {'Model':<20s} | {'R2':>8s} | {'MAE':>8s}")
print(f"  {'-'*20} | {'-'*8} | {'-'*8}")
print(f"  {'Random Forest':<20s} | {rf_r2:8.4f} | {rf_mae:8.2f}")
print(f"  {'LSTM':<20s} | {lstm_r2:8.4f} | {lstm_mae:8.2f}")
print(f"\n  Δ R2 (LSTM - RF):  {lstm_r2 - rf_r2:+.4f}")
print(f"  Δ MAE (LSTM - RF): {lstm_mae - rf_mae:+.2f}")

# Save results
results = {
    'rf': {'r2': float(rf_r2), 'mae': float(rf_mae)},
    'lstm': {'r2': float(lstm_r2), 'mae': float(lstm_mae)},
    'delta_r2': float(lstm_r2 - rf_r2),
    'delta_mae': float(lstm_mae - rf_mae),
    'config': {
        'epochs': EPOCHS,
        'hidden_size': 64,
        'num_layers': 2,
        'batch_size': 512,
        'lr': 0.001,
        'features': len(base_features),
        'device': str(device),
    }
}

out = RESULTS_DIR / "lstm_vs_rf.json"
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved: {out}")
print("\n✅ LSTM experiment complete!")
