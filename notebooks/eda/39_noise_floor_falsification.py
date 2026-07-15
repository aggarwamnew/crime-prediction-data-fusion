"""39_noise_floor_falsification.py — Why the noise-floor RATIO is not a floor test.

The binned "noise floor" ratio (actual_MAE / sqrt(2*var_residual/pi)) equals (MAD/SD)*sqrt(pi/2),
a SCALE-INVARIANT function of the residual distribution's shape: it is ~1 for any roughly-symmetric
residuals and is blind to the residual magnitude (the quantity a floor claim is about). This script
falsifies the "model has reached the noise floor" reading by feeding models of very different quality
into the same machinery: reasonable models (full RF, lag-1, per-unit mean) all score ~0.94-0.99
despite different MAE, so the ratio does not measure floor proximity. Basis for softening the
noise-floor claims to a residual-symmetry observation (see results/discussion noise-floor sections).
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
sys.path.insert(0,'.')
from src.data.db import ThesisDB
db=ThesisDB()
mc=db.query("SELECT lsoa_code, month, COUNT(*) crime_count FROM crime_clean GROUP BY lsoa_code, month")
tot=mc.groupby('lsoa_code')['crime_count'].sum(); active=tot[tot>=36].index; mc=mc[mc.lsoa_code.isin(active)]
months=sorted(mc.month.unique()); lsoas=sorted(mc.lsoa_code.unique())
grid=pd.MultiIndex.from_product([lsoas,months],names=['lsoa_code','month'])
df=mc.set_index(['lsoa_code','month']).reindex(grid,fill_value=0).reset_index().sort_values(['lsoa_code','month'])
for lag in [1,2,3,6,12]: df[f'lag_{lag}']=df.groupby('lsoa_code')['crime_count'].shift(lag)
for w in [3,6,12]: df[f'rm_{w}']=df.groupby('lsoa_code')['crime_count'].transform(lambda x:x.shift(1).rolling(w,min_periods=1).mean())
df['month_sin']=np.sin(2*np.pi*pd.to_datetime(df.month).dt.month/12); df['month_cos']=np.cos(2*np.pi*pd.to_datetime(df.month).dt.month/12)
df['time_idx']=df.month.map({m:i for i,m in enumerate(months)})
BASE=['lag_1','lag_2','lag_3','lag_6','lag_12','rm_3','rm_6','rm_12','month_sin','month_cos','time_idx']
df=df.dropna(subset=BASE)
tmv=months[-6:]; tr=df[~df.month.isin(tmv)]; te=df[df.month.isin(tmv)].copy()

def ratio_report(name, pred):
    t=te.copy(); t['pred']=pred; t['res']=t.crime_count-t['pred']
    mae=mean_absolute_error(t.crime_count,t['pred']); r2=r2_score(t.crime_count,t['pred'])
    # bin by predicted rank into deciles
    t['b']=pd.qcut(t['pred'].rank(method='first'),10,labels=False)
    bs=t.groupby('b').agg(mp=('pred','mean'),vr=('res','var'),amae=('res',lambda x:x.abs().mean())).reset_index()
    bs['theory']=np.sqrt(2*bs.vr/np.pi); bs['ratio']=bs.amae/bs.theory
    r19=bs.ratio.iloc[:9]
    print(f"{name:32s} R2={r2:6.3f} MAE={mae:6.3f} | floor-ratio(bins1-9) mean={r19.mean():.2f} [{r19.min():.2f},{r19.max():.2f}]")

# 1. Full RF baseline (the thesis model)
rf=RandomForestRegressor(n_estimators=200,max_depth=15,min_samples_leaf=5,n_jobs=-1,random_state=42); rf.fit(tr[BASE],tr.crime_count)
ratio_report("Full RF baseline (R2~0.94)", rf.predict(te[BASE]))
# 2. Deliberately weak: lag-1 only
ratio_report("Weak: predict = lag_1", te['lag_1'].values)
# 3. Much weaker: predict each LSOA's train mean
lm=tr.groupby('lsoa_code').crime_count.mean(); ratio_report("Weak: per-LSOA train mean", te.lsoa_code.map(lm).values)
# 4. Terrible: predict global constant (mean)
ratio_report("Terrible: global constant", np.full(len(te), tr.crime_count.mean()))
# 5. Add noise to the good model (degrade it)
rng=np.random.default_rng(0); ratio_report("RF + large gaussian noise", rf.predict(te[BASE])+rng.normal(0,8,len(te)))
