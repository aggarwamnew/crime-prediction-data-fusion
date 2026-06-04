"""
Crime Prediction Explorer — Interactive Streamlit App (v3)

Shows predictions + feature overlay side-by-side across London's LSOAs.
Time slider for 6 test months. Actual/Predicted/Residual views.

Usage: streamlit run src/visualization/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from src.utils.config import PROJECT_ROOT

st.set_page_config(
    page_title="London Crime Prediction Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div { padding-top: 0.5rem; }
    h1 { color: #1a1a2e; font-size: 1.8rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.7rem 0.8rem; border-radius: 10px; color: white; text-align: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.7rem 0.8rem; border-radius: 10px; color: white; text-align: center;
    }
    .metric-card-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 0.7rem 0.8rem; border-radius: 10px; color: white; text-align: center;
    }
    .metric-card h3, .metric-card-green h3, .metric-card-red h3 { margin: 0; font-size: 1.4rem; }
    .metric-card p, .metric-card-green p, .metric-card-red p { margin: 0; font-size: 0.7rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_predictions():
    return pd.read_parquet(PROJECT_ROOT / "data/processed/london/predictions.parquet")

@st.cache_data
def load_features():
    return gpd.read_parquet(PROJECT_ROOT / "data/processed/london/lsoa_features.parquet")

@st.cache_data
def get_geojson(_gdf):
    return json.loads(_gdf.to_json())


try:
    predictions = load_predictions()
    features_gdf = load_features()
except Exception as e:
    st.error(f"Data not found. Run `python src/visualization/precompute_data.py` first.\n\n{e}")
    st.stop()

geojson = get_geojson(features_gdf)
crime_types = sorted(predictions['crime_type'].unique())
months = sorted(predictions['month'].unique())
month_labels = {m: pd.to_datetime(m).strftime('%b %Y') for m in months}

FEATURE_LAYERS = {}
feature_map = {
    "IMD Score": "imd_score",
    "Population Density": "pop_density",
    "Bus Stops": "poi_bus_stops",
    "Restaurants": "poi_restaurants",
    "Pubs": "poi_pubs",
    "Total POIs": "poi_total",
    "Shops": "poi_shops",
    "House Price": "median_house_price",
    "Mental Health": "samhi_index",
}
for k, v in feature_map.items():
    if v in features_gdf.columns:
        FEATURE_LAYERS[k] = v

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 🔍 Controls")
    st.markdown("---")

    crime_type = st.selectbox("**Crime Type**", crime_types,
                               index=crime_types.index('All crimes') if 'All crimes' in crime_types else 0)

    view_mode = st.radio("**Prediction View**", ["Predicted", "Actual", "Residual"])

    st.markdown("---")
    month_idx = st.slider("**Month**", 0, len(months) - 1, 0)
    selected_month = months[month_idx]
    st.markdown(f"### 📅 {month_labels[selected_month]}")

    st.markdown("---")
    show_overlay = st.checkbox("**Show Feature Overlay**", value=True)
    if show_overlay:
        feature_label = st.selectbox("**Feature**", list(FEATURE_LAYERS.keys()))

    st.markdown("---")
    clip_pct = st.slider("**Color clip %ile**", 85, 100, 95)

# ── Data ──
month_data = predictions[
    (predictions['crime_type'] == crime_type) &
    (predictions['month'] == selected_month)
].copy()

# ── Title + Metrics ──
st.markdown("# 🗺️ London Crime Prediction Explorer")

if len(month_data) > 0:
    mae = month_data['residual'].abs().mean()
    ss_res = (month_data['residual'] ** 2).sum()
    ss_tot = ((month_data['crime_count'] - month_data['crime_count'].mean()) ** 2).sum()
    r2_month = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>{month_data["crime_count"].sum():,.0f}</h3><p>Total Actual</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card-green"><h3>{month_data["predicted"].sum():,.0f}</h3><p>Total Predicted</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card-red"><h3>{mae:.2f}</h3><p>MAE</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>{r2_month:.3f}</h3><p>R²</p></div>', unsafe_allow_html=True)

st.markdown("")

# ── Map column selection ──
if view_mode == "Predicted":
    map_col, color_label, color_scale = 'predicted', 'Predicted crimes', 'YlOrRd'
elif view_mode == "Actual":
    map_col, color_label, color_scale = 'crime_count', 'Actual crimes', 'YlOrRd'
else:
    map_col, color_label, color_scale = 'residual', 'Actual − Predicted', 'RdBu'

# ── Maps layout ──
def make_map(data, col, label, cscale, vmin=None, vmax=None):
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = data[col].quantile(clip_pct / 100)
        if vmax == vmin:
            vmax = data[col].max() or 1

    fig = px.choropleth_map(
        data, geojson=geojson, locations='lsoa_code',
        featureidkey='properties.lsoa_code',
        color=col, color_continuous_scale=cscale,
        range_color=[vmin, vmax],
        zoom=9.2, center={"lat": 51.509, "lon": -0.118},
        opacity=0.7, hover_name='lsoa_code',
        hover_data={col: ':.2f'}, labels={col: label},
    )
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=520,
        coloraxis_colorbar=dict(title=dict(text=label, font=dict(size=10)), thickness=10, len=0.5),
    )
    return fig

pred_data = month_data[['lsoa_code', map_col]].copy()

if view_mode == "Residual":
    abs_max = max(abs(pred_data[map_col].quantile(0.02)), abs(pred_data[map_col].quantile(0.98)), 1)
    pred_fig = make_map(pred_data, map_col, color_label, color_scale, -abs_max, abs_max)
else:
    pred_fig = make_map(pred_data, map_col, color_label, color_scale)

if show_overlay:
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"**{view_mode}** · {crime_type} · {month_labels[selected_month]}")
        st.plotly_chart(pred_fig, width='stretch')
    with col_right:
        feat_col = FEATURE_LAYERS[feature_label]
        feat_data = features_gdf[['lsoa_code', feat_col]].dropna(subset=[feat_col])
        feat_fig = make_map(feat_data, feat_col, feature_label, 'Viridis')
        st.markdown(f"**{feature_label}** overlay")
        st.plotly_chart(feat_fig, width='stretch')
else:
    st.markdown(f"**{view_mode}** · {crime_type} · {month_labels[selected_month]}")
    st.plotly_chart(pred_fig, width='stretch')

# ── Trend charts ──
st.markdown("### 📊 6-Month Trend")
monthly_agg = predictions[predictions['crime_type'] == crime_type].groupby('month').agg(
    actual=('crime_count', 'sum'),
    predicted=('predicted', 'sum'),
    mae=('residual', lambda x: x.abs().mean()),
).reset_index()
monthly_agg['month_label'] = monthly_agg['month'].map(month_labels)

trend_left, trend_right = st.columns(2)
with trend_left:
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=monthly_agg['month_label'], y=monthly_agg['actual'],
                                name='Actual', marker_color='#667eea', opacity=0.8))
    fig_trend.add_trace(go.Bar(x=monthly_agg['month_label'], y=monthly_agg['predicted'],
                                name='Predicted', marker_color='#38ef7d', opacity=0.8))
    fig_trend.update_layout(barmode='group', height=250,
                             margin={"r": 10, "t": 10, "l": 10, "b": 40},
                             legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                             yaxis_title="Total Crimes")
    st.plotly_chart(fig_trend, width='stretch')

with trend_right:
    fig_mae = go.Figure()
    fig_mae.add_trace(go.Scatter(x=monthly_agg['month_label'], y=monthly_agg['mae'],
                                  mode='lines+markers', line=dict(color='#eb3349', width=2),
                                  marker=dict(size=8)))
    fig_mae.update_layout(height=250, margin={"r": 10, "t": 10, "l": 10, "b": 40},
                           yaxis_title="MAE per LSOA")
    st.plotly_chart(fig_mae, width='stretch')

st.markdown("---")
st.markdown("*MSc Thesis — Crime Prediction with Data Fusion · Trinity College Dublin*")
