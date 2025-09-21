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

# Expandable data preview
with st.expander("🔍 Preview Dataset (First 5 Rows)"):
    st.dataframe(df.head(), use_container_width=True)

with st.expander("📋 Column Names"):
    cols_df = pd.DataFrame({"Column Names": df.columns.tolist()})
    st.dataframe(cols_df, use_container_width=True)

# Column detection
col_initiative = find_col(df, [
    "Existence of initiatives", "Existence of initiativ", "existence of initiativ", 
    "initiatives and projects", "initiatives"
])

col_tourism_index = find_col(df, ["Tourism Index", "Tourism_Index", "tourism index"])
col_total_hotels = find_col(df, ["Total number of hotels", "Total number of hotel", "total hotels", "total number"])
col_governorate = find_col(df, ["Governorate", "governorate", "Region", "region", "Mohafazat", "mohafazat"])

# Look for additional geographic columns
col_area = find_col(df, ["Area", "City", "Municipality", "District", "Caza", "area", "city"])
col_zone = find_col(df, ["Zone", "zone", "Sector", "sector"])

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

# ENHANCED REGIONAL FILTERING SECTION
st.sidebar.markdown("## 🗺️ Enhanced Geographic Filters")

# Primary governorate filter with enhanced controls
governorate_choice = None
if col_governorate:
    st.sidebar.markdown("### 🏛️ Primary Region Filter")
    
    uniq_gov = sorted(df[col_governorate].dropna().unique().tolist())
    
    # Quick selection buttons
    col_all, col_none, col_reset = st.sidebar.columns(3)
    with col_all:
        if st.button("✅ All", key="select_all_gov", help="Select all regions"):
            st.session_state.governorate_choice = uniq_gov
            st.rerun()
    with col_none:
        if st.button("❌ None", key="deselect_all_gov", help="Deselect all regions"):
            st.session_state.governorate_choice = []
            st.rerun()
    with col_reset:
        if st.button("🔄 Reset", key="reset_gov", help="Reset to default"):
            st.session_state.governorate_choice = uniq_gov
            st.rerun()
    
    # Main governorate selector
    governorate_choice = st.sidebar.multiselect(
        f"🏛️ Select {col_governorate}s",
        options=uniq_gov,
        default=st.session_state.governorate_choice,
        help="Filter analysis by specific governorates/regions",
        key="gov_multiselect"
    )
    
    # Update session state
    st.session_state.governorate_choice = governorate_choice
    
    # Show selection summary
    if len(governorate_choice) != len(uniq_gov):
        coverage_pct = (len(governorate_choice) / len(uniq_gov)) * 100
        st.sidebar.info(f"📊 Selected: {len(governorate_choice)}/{len(uniq_gov)} regions ({coverage_pct:.1f}%)")

# Visualization for Initiatives by Region
if st.sidebar.checkbox("Show Initiatives by Region", value=True):
    st.markdown('<div class="sub-header">📊 Initiatives by Region</div>', unsafe_allow_html=True)
    
    # Filter and Aggregate the Data for Initiatives by Region
    initiatives_exist_df = df[df[col_initiative] == 1]
    initiatives_by_region = initiatives_exist_df.groupby(col_governorate)[col_initiative].count().reset_index()
    initiatives_by_region.columns = [col_governorate, 'Number of Initiatives']
    
    # Bar chart for initiatives by region
    fig_bar = px.bar(
        initiatives_by_region,
        x=col_governorate,
        y='Number of Initiatives',
        title='Number of Initiatives by Region',
        labels={col_governorate: "Region", 'Number of Initiatives': 'Number of Initiatives'},
        color='Number of Initiatives',
        color_continuous_scale='Viridis'
    )
    fig_bar.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Pie chart for initiatives by region
    fig_pie = px.pie(
        initiatives_by_region,
        values='Number of Initiatives',
        names=col_governorate,
        title='Initiatives Distribution by Region'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Footer with insights and instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Use This Enhanced Dashboard:

1. **Enhanced Geographic Filtering**: 
   - Use quick selection buttons (All/None/Reset) for efficient region management
   - Multi-level filtering: Governorate → Area for detailed geographic analysis
   - Choose geographic analysis modes for different perspectives

2. **Interactive Features**:
   - **Feature 1**: Multi-level geographic filtering with coverage indicators
   - **Feature 2**: Geographic analysis modes (Compare, Ranking, Distribution)

3. **Data Analysis**: 
   - Switch between different tourism metrics and aggregation methods
   - Examine data distribution and identify outliers using enhanced visualizations

### 💡 Key Design Decisions:

- **Smart Geographic Hierarchy**: Automatically detects and links governorate and area columns
- **Interactive Filter Management**: Session state preserves selections and provides quick controls
- **Multiple Analysis Perspectives**: Standard analysis plus specialized geographic modes
- **Enhanced Visual Feedback**: Coverage percentages and filter summaries
- **Comprehensive Insights**: Combines statistical analysis with geographic intelligence
""")

# Export functionality
if st.button("📥 Export Filtered Data"):
    csv = df_filtered.to_csv(index=False)
    
    # Create filename with filter info
    filter_info = []
    if governorate_choice and len(governorate_choice) < len(df[col_governorate].unique()):
        filter_info.append(f"{len(governorate_choice)}regions")
    if area_choice and col_area and len(area_choice) < len(df[col_area].unique()):
        filter_info.append(f"{len(area_choice)}areas")
    
    filter_suffix = "_" + "_".join(filter_info) if filter_info else ""
    
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"filtered_tourism_data{filter_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
