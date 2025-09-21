import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("🎈 My new Streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# app.py
st.set_page_config(page_title="Tourism Visualizations - Interactive", layout="wide")

st.title("Interactive Tourism Visualizations")
st.write(
    """
This app reproduces two interactive visualizations.
Use the sidebar controls to filter and explore the data.
"""
)

# Default dataset URL used in your Colab notebook
DEFAULT_CSV_URL = "https://linked.aub.edu.lb/pkgcube/data/551015b5649368dd2612f795c2a9c2d8_20240902_115953.csv"

@st.cache_data
def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)

def find_col(df, candidates):
    """
    Return first column in df whose name contains any of the candidate substrings (case-insensitive).
    """
    if df is None:
        return None
    cols = df.columns.tolist()
    lowmap = {c.lower(): c for c in cols}
    for cand in candidates:
        cand_l = cand.lower()
        # exact or substring match
        for original in cols:
            if cand_l in original.lower():
                return original
    return None

# Load data (try URL first, then allow manual upload)
st.sidebar.header("Data source")
use_url = st.sidebar.checkbox("Load dataset from the original URL", value=True)
df = None
err = None
if use_url:
    st.sidebar.caption("If the URL is inaccessible, upload the CSV below.")
    df, err = load_data_from_url(DEFAULT_CSV_URL)
    if err:
        st.sidebar.error(f"Could not load from URL: {err}")

if df is None:
    uploaded = st.sidebar.file_uploader("Or upload the CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {e}")

if df is None:
    st.warning("No data loaded yet. Toggle the sidebar option to upload a CSV or enable URL loading.")
    st.stop()

st.sidebar.success(f"Loaded dataset with {df.shape[0]:,} rows and {df.shape[1]} columns")

# Show a small sample and column list (collapsible)
with st.expander("Preview data (first 5 rows)"):
    st.dataframe(df.head())

with st.expander("Column names"):
    st.write(list(df.columns))

# Attempt to detect the notebook's important columns by substring
col_initiative = find_col(df, [
    "Existence of initiatives",
    "Existence of initiativ",
    "existence of initiativ",
    "initiatives and projects"
])

col_tourism_index = find_col(df, ["Tourism Index", "Tourism_Index", "tourism index"])
col_total_hotels = find_col(df, ["Total number of hotels", "Total number of hotel", "total hotels", "total number"])

# Let user pick a numeric metric if we detected several numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
# Include some named candidates at top if present
preferred_metrics = []
if col_tourism_index:
    preferred_metrics.append(col_tourism_index)
if col_total_hotels:
    preferred_metrics.append(col_total_hotels)
# add other numeric cols but keep unique
for c in numeric_cols:
    if c not in preferred_metrics:
        preferred_metrics.append(c)

st.sidebar.header("Interactive controls")

# ENHANCED REGIONAL FILTERING SECTION - INSERT HERE
# Governorate / Region filter if present
col_governorate = find_col(df, ["Governorate", "governorate", "Region", "region", "Mohafazat", "mohafazat"])

# Initialize session state for region selection if not exists
if 'governorate_choice' not in st.session_state:
    if col_governorate:
        st.session_state.governorate_choice = sorted(df[col_governorate].dropna().unique().tolist())
    else:
        st.session_state.governorate_choice = []

governorate_choice = None
if col_governorate:
    st.sidebar.markdown("## 🗺️ Geographic Filters")
    
    uniq_gov = sorted(df[col_governorate].dropna().unique().tolist())
    
    # Select All/None buttons
    col_all, col_none = st.sidebar.columns(2)
    with col_all:
        if st.button("✅ All", key="select_all", help="Select all regions"):
            st.session_state.governorate_choice = uniq_gov
    with col_none:
        if st.button("❌ None", key="deselect_all", help="Deselect all regions"):
            st.session_state.governorate_choice = []
    
    # Main region selector
    governorate_choice = st.sidebar.multiselect(
        f"🏛️ Select {col_governorate}s",
        options=uniq_gov,
        default=st.session_state.governorate_choice,
        help="Filter analysis by specific governorates/regions"
    )
    
    # Update session state
    st.session_state.governorate_choice = governorate_choice
    
    # Show selection summary
    if len(governorate_choice) < len(uniq_gov):
        st.sidebar.info(f"📊 Selected: {len(governorate_choice)} of {len(uniq_gov)} regions")
    
    # Geographic analysis mode
    st.sidebar.markdown("### 🔍 Analysis Mode")
    geo_analysis_mode = st.sidebar.selectbox(
        "Geographic Focus:",
        ["Standard Analysis", "Compare Regions", "Regional Ranking"],
        help="Choose how to analyze the geographic data"
    )

# Initiative filter (multi-select)
if col_initiative:
    uniq_init = sorted(df[col_initiative].dropna().unique().tolist())
    selected_initiatives = st.sidebar.multiselect("Initiative status (filter)", options=uniq_init, default=uniq_init)
else:
    # fallback: allow user to pick any categorical column to group by
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    selected_cat = st.sidebar.selectbox("No initiative column detected – choose a categorical column to group by", options=[None] + categorical_cols, index=0)
    if selected_cat:
        col_initiative = selected_cat
        uniq_init = sorted(df[col_initiative].dropna().unique().tolist())
        selected_initiatives = st.sidebar.multiselect("Values (filter)", options=uniq_init, default=uniq_init)
    else:
        selected_initiatives = None

# Metric selector
metric = st.sidebar.selectbox("Select numeric metric to analyze", preferred_metrics if preferred_metrics else numeric_cols)

# Optional: choose aggregation
agg_func = st.sidebar.selectbox("Aggregation for visualization 1", ["mean", "median", "sum", "count"], index=0)

# Apply filters to dataframe
df_filtered = df.copy()
if governorate_choice is not None and len(governorate_choice) > 0:
    df_filtered = df_filtered[df_filtered[col_governorate].isin(governorate_choice)]
if col_initiative and selected_initiatives is not None:
    df_filtered = df_filtered[df_filtered[col_initiative].isin(selected_initiatives)]

# Enhanced geographic metrics display
if col_governorate and governorate_choice is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df_filtered)
        st.metric("📊 Filtered Records", f"{total_records:,}")
    
    with col2:
        regions_count = len(governorate_choice) if governorate_choice else 0
        st.metric("🗺️ Selected Regions", regions_count)
    
    with col3:
        if metric in df_filtered.columns and len(df_filtered) > 0:
            avg_metric = df_filtered[metric].mean()
            st.metric(f"📈 Avg {metric}", f"{avg_metric:.2f}" if not pd.isna(avg_metric) else "N/A")
    
    with col4:
        if len(df_filtered) > 0 and col_governorate:
            coverage_pct = (len(governorate_choice) / len(df[col_governorate].dropna().unique())) * 100
            st.metric("📈 Coverage", f"{coverage_pct:.1f}%")

st.markdown("## Visualization 1 – Aggregated metric by initiative status")
st.write(
    "This visualization shows an aggregate (mean/median/sum/count) of the selected metric grouped by initiative status (or chosen categorical column)."
)

# Compute aggregation
if col_initiative:
    if agg_func == "mean":
        agg_df = df_filtered.groupby(col_initiative)[metric].mean().reset_index().sort_values(by=metric, ascending=False)
    elif agg_func == "median":
        agg_df = df_filtered.groupby(col_initiative)[metric].median().reset_index().sort_values(by=metric, ascending=False)
    elif agg_func == "sum":
        agg_df = df_filtered.groupby(col_initiative)[metric].sum().reset_index().sort_values(by=metric, ascending=False)
    else:  # count
        agg_df = df_filtered.groupby(col_initiative)[metric].count().reset_index().sort_values(by=metric, ascending=False)
else:
    st.error("No categorical column (initiative) detected to group by.")
    st.stop()

fig1 = px.bar(agg_df, x=col_initiative, y=metric, title=f"{agg_func.title()} of {metric} by {col_initiative}",
              labels={col_initiative: col_initiative, metric: metric})
fig1.update_layout(xaxis={'categoryorder':'total descending'})
st.plotly_chart(fig1, use_container_width=True)

# ENHANCED GEOGRAPHIC ANALYSIS SECTION
if col_governorate and governorate_choice and 'geo_analysis_mode' in locals():
    if geo_analysis_mode == "Compare Regions":
        st.markdown("### 🏛️ Regional Comparison")
        
        if len(governorate_choice) > 1:
            # Create regional comparison
            regional_stats = df_filtered.groupby(col_governorate)[metric].agg(['mean', 'count']).round(2)
            regional_stats.columns = ['Average', 'Count']
            
            # Horizontal bar chart for better region name visibility
            fig_regional = px.bar(
                regional_stats.reset_index(), 
                x='Average', 
                y=col_governorate,
                orientation='h',
                title=f"Regional Comparison: Average {metric}",
                color='Average',
                color_continuous_scale="viridis"
            )
            fig_regional.update_layout(height=max(400, len(governorate_choice) * 50))
            st.plotly_chart(fig_regional, use_container_width=True)
        else:
            st.info("Select multiple regions to see comparison")
    
    elif geo_analysis_mode == "Regional Ranking":
        st.markdown("### 🏆 Regional Performance Ranking")
        
        if metric and len(governorate_choice) > 1:
            ranking_df = df_filtered.groupby(col_governorate).agg({
                metric: ['mean', 'count']
            }).round(2)
            
            ranking_df.columns = ['Average', 'Count']
            ranking_df = ranking_df.sort_values('Average', ascending=False).reset_index()
            ranking_df['Rank'] = range(1, len(ranking_df) + 1)
            
            # Reorder columns
            ranking_df = ranking_df[['Rank', col_governorate, 'Average', 'Count']]
            
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        else:
            st.info("Select multiple regions to see ranking")

# Visualization 2
st.markdown("## Visualization 2 – Distribution of 'Total number of hotels' (box plot)")
st.write(
    "This visualization reproduces the box plot from your Colab: distribution of total number of hotels grouped by initiative status.\n"
    "If the original 'Total number of hotels' column is not detected, you may choose another numeric column."
)

# Choose column to show distribution
dist_candidates = []
if col_total_hotels:
    dist_candidates.append(col_total_hotels)
# include metric as alternative
if metric not in dist_candidates:
    dist_candidates.append(metric)
# include other numeric options
for n in numeric_cols:
    if n not in dist_candidates:
        dist_candidates.append(n)

dist_col = st.selectbox("Select numeric column for distribution (box plot)", dist_candidates, index=0)

# Prepare box plot data
if col_initiative:
    # ensure the dist_col is numeric
    df_box = df_filtered[[col_initiative, dist_col]].dropna()
    # cast to numeric if possible
    try:
        df_box[dist_col] = pd.to_numeric(df_box[dist_col], errors='coerce')
    except Exception:
        pass
    df_box = df_box.dropna(subset=[dist_col])
    if df_box.empty:
        st.warning("No data available for the chosen distribution column and filters.")
    else:
        fig2 = px.box(df_box, x=col_initiative, y=dist_col,
                      title=f"Distribution of {dist_col} by {col_initiative}",
                      labels={col_initiative: col_initiative, dist_col: dist_col})
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.error("Need a categorical column to group distributions by.")

# Enhanced insights section
with st.expander("🔍 Enhanced Data Insights"):
    st.write(f"**Analysis Summary (Filtered Data):**")
    st.write(f"- Total rows: {len(df_filtered):,}")
    
    if col_governorate and governorate_choice:
        st.write(f"- Regions analyzed: {len(governorate_choice)}")
        
        # Top performing region
        if len(df_filtered) > 0:
            regional_avg = df_filtered.groupby(col_governorate)[metric].mean()
            top_region = regional_avg.idxmax()
            top_value = regional_avg.max()
            st.write(f"- Top performing region: **{top_region}** ({top_value:.2f})")
    
    # Show top groups
    st.write("**Top performing groups:**")
    st.dataframe(agg_df.head(5).reset_index(drop=True))

st.markdown("---")
st.caption("Enhanced with geographic filtering and regional analysis capabilities")
