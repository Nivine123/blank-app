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
.context-box {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
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

# Expandable data preview
with st.expander("🔍 Preview Dataset (First 5 Rows)"):
    st.dataframe(df.head(), use_container_width=True)

with st.expander("📋 Column Names"):
    cols_df = pd.DataFrame({"Column Names": df.columns.tolist()})
    st.dataframe(cols_df, use_container_width=True)

# Column detection (use explicit column names if find_col fails)
col_initiative = find_col(df, [
    "Existence of initiatives", "Existence of initiativ", "existence of initiativ", 
    "initiatives and projects", "initiatives"
])

# If find_col failed, manually set the column name based on the dataset
if not col_initiative:
    col_initiative = 'Existence of initiatives and projects in the past five years to improve the tourism sector - exists'

col_tourism_index = find_col(df, ["Tourism Index", "Tourism_Index", "tourism index"])
col_total_hotels = find_col(df, ["Total number of hotels", "Total number of hotel", "total hotels", "total number"])
col_governorate = find_col(df, ["Governorate", "governorate", "Region", "region", "Mohafazat", "mohafazat"])

# Numeric columns for analysis
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
preferred_metrics = []
if col_tourism_index and col_tourism_index in numeric_cols:
    preferred_metrics.append(col_tourism_index)
if col_total_hotels and col_total_hotels in numeric_cols:
    preferred_metrics.append(col_total_hotels)
for c in numeric_cols:
    if c not in preferred_metrics:
        preferred_metrics.append(c)

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

# Apply filters
df_filtered = df.copy()

# Apply governorate filter
if col_governorate:
    governorate_choice = st.sidebar.multiselect(
        "Select Governorate",
        options=df[col_governorate].dropna().unique(),
        default=st.session_state.governorate_choice
    )
    df_filtered = df_filtered[df_filtered[col_governorate].isin(governorate_choice)]

# Apply initiative filter
df_initiatives = df_filtered[df_filtered[col_initiative] == 1]  # Filter based on initiatives

# Ensure col_governorate exists in the dataframe after filtering
if col_governorate in df_initiatives.columns:
    # Count initiatives by region
    initiative_counts_by_region = df_initiatives[col_governorate].value_counts().reset_index()
    initiative_counts_by_region.columns = ['Region', 'Number of Initiatives']

    # Calculate average tourism index by region
    tourism_index_by_region = df_initiatives.groupby(col_governorate)[col_tourism_index].mean().reset_index()
    tourism_index_by_region.columns = ['Region', 'Average Tourism Index']

    # Merge the two dataframes
    merged_df = pd.merge(initiative_counts_by_region, tourism_index_by_region, on='Region')

    # Bar chart: Number of initiatives and average tourism index by region
    fig = px.bar(
        merged_df,
        x='Region',
        y='Number of Initiatives',
        title='Number of Tourism Initiatives and Average Tourism Index by Region',
        labels={'Region': 'Region', 'Number of Initiatives': 'Number of Initiatives'},
        color='Average Tourism Index',
        color_continuous_scale='Viridis',
        hover_data=['Average Tourism Index']
    )

    # Update layout for better readability
    fig.update_layout(
        barmode='group',
        xaxis={'tickangle': 45},  # Rotate region names
        showlegend=False
    )

    # Show the plot
    st.plotly_chart(fig)

else:
    st.error(f"Column {col_governorate} not found in the data.")
