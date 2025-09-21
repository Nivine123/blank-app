import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Tourism Analytics Dashboard", 
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #1e88e5;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #e53935;
    margin: 1.5rem 0 1rem 0;
}
.insight-box {
    background-color: #e3f2fd;
    border-left: 5px solid #1e88e5;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
.metric-card {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
}
.context-box {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
.geographic-box {
    background-color: #f3e5f5;
    border-left: 5px solid #9c27b0;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏨 Interactive Tourism Analytics Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>📊 Dashboard Overview</strong><br>
This interactive dashboard analyzes tourism data across different regions and initiative types. 
The visualizations help identify patterns in tourism development and infrastructure distribution.
Use the enhanced sidebar controls to filter data and explore different geographic perspectives.
</div>
""", unsafe_allow_html=True)

# Default dataset URL
DEFAULT_CSV_URL = "https://linked.aub.edu.lb/pkgcube/data/551015b5649368dd2612f795c2a9c2d8_20240902_115953.csv"

@st.cache_data
def load_data_from_url(url):
    try:
        df = pd.read_csv(url)
        return df, None
    except Exception as e:
        return None, str(e)

def find_col(df, candidates):
    """Return first column in df whose name contains any of the candidate substrings (case-insensitive)."""
    if df is None:
        return None
    cols = df.columns.tolist()
    for cand in candidates:
        cand_l = cand.lower()
        for original in cols:
            if cand_l in original.lower():
                return original
    return None

# Sidebar for data loading and controls
st.sidebar.markdown("## 📁 Data Source")
use_url = st.sidebar.checkbox("Load dataset from URL", value=True)

df = None
err = None

if use_url:
    st.sidebar.caption("📡 Loading from online source...")
    df, err = load_data_from_url(DEFAULT_CSV_URL)
    if err:
        st.sidebar.error(f"❌ Could not load from URL: {err}")

if df is None:
    uploaded = st.sidebar.file_uploader("📂 Upload CSV file", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("✅ File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")

if df is None:
    st.warning("⚠️ No data loaded yet. Please enable URL loading or upload a CSV file using the sidebar.")
    st.stop()

# Data info
st.sidebar.markdown("## 📊 Dataset Information")
st.sidebar.metric("Total Rows", f"{df.shape[0]:,}")
st.sidebar.metric("Total Columns", f"{df.shape[1]}")

# Column detection - FIXED based on actual data
col_initiative = find_col(df, [
    "Existence of initiatives and projects in the past five years to improve the tourism sector"
])

col_tourism_index = find_col(df, ["Tourism Index"])
col_total_hotels = find_col(df, ["Total number of hotels"])
col_governorate = find_col(df, ["refArea"])  # This contains the governorate URLs
col_town = find_col(df, ["Town"])  # This is the town/area name

# Initialize session state for regional filtering
if 'governorate_choice' not in st.session_state:
    if col_governorate:
        st.session_state.governorate_choice = sorted(df[col_governorate].dropna().unique().tolist())
    else:
        st.session_state.governorate_choice = []

if 'area_choice' not in st.session_state:
    if col_area:
        st.session_state.area_choice = sorted(df[col_area].dropna().unique().tolist())
    else:
        st.session_state.area_choice = []

# Geographic Analysis Mode
st.sidebar.markdown("### 🔍 Geographic Analysis Mode")
geo_analysis_mode = st.sidebar.selectbox(
    "Analysis Focus:",
    ["Standard Analysis", "Compare Regions", "Regional Ranking", "Geographic Distribution"],
    help="Choose how to analyze the geographic data"
)

# Ensure that geo_analysis_mode is defined before proceeding with the if/elif checks
if geo_analysis_mode == "Compare Regions" and len(governorate_choice) > 1:
    st.markdown("""
    <div class="geographic-box">
    <strong>🔍 Regional Comparison Mode:</strong> Comparing performance across selected regions
    </div>
    """, unsafe_allow_html=True)
    
    # Regional comparison analysis
    regional_stats = df_filtered.groupby(col_governorate)[metric].agg(['mean', 'count', 'std']).round(2)
    regional_stats.columns = ['Average', 'Count', 'Std Dev']
    
    # Create horizontal bar chart for better region name visibility
    fig_regional = px.bar(
        regional_stats.reset_index(), 
        x='Average', 
        y=col_governorate,
        orientation='h',
        title=f"Regional Comparison: Average {metric}",
        color='Average',
        color_continuous_scale="viridis",
        text='Average'
    )
    fig_regional.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_regional.update_layout(height=max(400, len(governorate_choice) * 60))
    st.plotly_chart(fig_regional, use_container_width=True)
    
    # Show detailed comparison table
    st.markdown("#### 📊 Detailed Regional Statistics")
    st.dataframe(regional_stats, use_container_width=True)

elif geo_analysis_mode == "Regional Ranking":
    st.markdown("""
    <div class="geographic-box">
    <strong>🏆 Regional Ranking Mode:</strong> Performance ranking of selected regions
    </div>
    """, unsafe_allow_html=True)
    
    if len(governorate_choice) > 1:
        ranking_df = df_filtered.groupby(col_governorate).agg({
            metric: ['mean', 'sum', 'count']
        }).round(2)
        
        ranking_df.columns = ['Average', 'Total', 'Count']
        ranking_df = ranking_df.sort_values('Average', ascending=False).reset_index()
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        # Add performance indicators
        ranking_df['Performance'] = pd.cut(ranking_df['Average'], 
                                         bins=3, 
                                         labels=['🔴 Below Average', '🟡 Average', '🟢 Above Average'])
        
        # Reorder columns
        ranking_df = ranking_df[['Rank', col_governorate, 'Average', 'Total', 'Count', 'Performance']]
        
        # Display the ranking data
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    else:
        st.info("🔍 Select multiple regions to see ranking comparison")
