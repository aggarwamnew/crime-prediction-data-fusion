"""02_lstm_full_fusion.py — LSTM vs RF on all 51 features. ISOLATED."""
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, torch.nn as nn, numpy as np, json, os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from data_loader import load_full_fusion_data
from src.utils.config import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "notebooks/experimental/results"
print("Loading all 51 features (read-only)...")
train, test, base_feats, all_feats = load_full_fusion_data()
print(f"Features: {len(all_feats)}, Train: {len(train):,}, Test: {len(test):,}")

# RF baseline
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf.fit(train[all_feats], train['crime_count'])
rf_r2 = r2_score(test['crime_count'], rf.predict(test[all_feats]))
rf_mae = mean_absolute_error(test['crime_count'], rf.predict(test[all_feats]))
print(f"  R2: {rf_r2:.4f}, MAE: {rf_mae:.2f}")

# LSTM
print("\n--- LSTM ---")
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_tr = torch.FloatTensor(scaler_X.fit_transform(train[all_feats].values)).unsqueeze(1)
X_te = torch.FloatTensor(scaler_X.transform(test[all_feats].values)).unsqueeze(1)
y_tr = torch.FloatTensor(scaler_y.fit_transform(train['crime_count'].values.reshape(-1,1)).ravel())

class LSTM(nn.Module):
    def __init__(self, n_in, h=128, nl=2): 
        super().__init__()
        self.lstm = nn.LSTM(n_in, h, nl, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(h,32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32,1))
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:,-1,:]).squeeze(-1)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = LSTM(len(all_feats), h=128).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=True)

for epoch in range(50):
    model.train()
    for bx, by in loader:
        opt.zero_grad(); loss = nn.MSELoss()(model(bx.to(device)), by.to(device)); loss.backward(); opt.step()
    if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}/50: loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    pred = scaler_y.inverse_transform(model(X_te.to(device)).cpu().numpy().reshape(-1,1)).ravel()
lstm_r2 = r2_score(test['crime_count'].values, pred)
lstm_mae = mean_absolute_error(test['crime_count'].values, pred)
print(f"  R2: {lstm_r2:.4f}, MAE: {lstm_mae:.2f}")

print(f"\n{'Model':<15s} | {'R2':>8s} | {'MAE':>8s}")
print(f"{'RF (51 feat)':<15s} | {rf_r2:8.4f} | {rf_mae:8.2f}")
print(f"{'LSTM (51 feat)':<15s} | {lstm_r2:8.4f} | {lstm_mae:8.2f}")
print(f"Delta R2: {lstm_r2-rf_r2:+.4f}")

with open(RESULTS_DIR/"lstm_full_fusion.json",'w') as f:
    json.dump({'rf':{'r2':float(rf_r2),'mae':float(rf_mae)},'lstm':{'r2':float(lstm_r2),'mae':float(lstm_mae)},'features':len(all_feats)}, f, indent=2)
print(f"\n✅ Saved: {RESULTS_DIR}/lstm_full_fusion.json")
