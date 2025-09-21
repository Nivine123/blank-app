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

# Show detected columns (debug info)
with st.expander("🔧 Column Detection Debug Info"):
    col_debug1, col_debug2 = st.columns(2)
    with col_debug1:
        st.write("**Detected Columns:**")
        st.write(f"- Initiative: {col_initiative}")
        st.write(f"- Tourism Index: {col_tourism_index}")  
        st.write(f"- Total Hotels: {col_total_hotels}")
        st.write(f"- Governorate: {col_governorate}")
        st.write(f"- Town/Area: {col_area}")
        
    with col_debug2:
        st.write("**Sample Data:**")
        if col_governorate and col_initiative:
            sample_data = df[[col_governorate, col_initiative, metric]].head()
            st.dataframe(sample_data)
        
        if col_governorate:
            st.write("**Available Governorates:**")
            st.write(list(df[col_governorate].unique()[:10]))

# Column detection - FIXED based on actual data
col_initiative = find_col(df, [
    "Existence of initiatives and projects in the past five years to improve the tourism sector"
])

col_tourism_index = find_col(df, ["Tourism Index"])
col_total_hotels = find_col(df, ["Total number of hotels"])
col_governorate = find_col(df, ["refArea"])  # This contains the governorate URLs
col_town = find_col(df, ["Town"])  # This is the town/area name

# Convert refArea URLs to readable governorate names
if col_governorate and col_governorate in df.columns:
    def extract_governorate_name(url):
        if pd.isna(url):
            return "Unknown"
        # Extract governorate name from URL
        if "Mount_Lebanon_Governorate" in str(url):
            return "Mount Lebanon"
        elif "North_Governorate" in str(url):
            return "North"
        elif "South_Governorate" in str(url):
            return "South"  
        elif "Beqaa_Governorate" in str(url):
            return "Beqaa"
        elif "Nabatieh_Governorate" in str(url):
            return "Nabatieh"
        elif "Akkar_Governorate" in str(url):
            return "Akkar"
        elif "Baalbek-Hermel_Governorate" in str(url):
            return "Baalbek-Hermel"
        else:
            # Try to extract district names for more granular analysis
            url_str = str(url)
            if "_District" in url_str:
                district = url_str.split("/")[-1].replace("_District", "").replace("_", " ")
                return district
            return "Other"
    
    # Create a clean governorate column
    df['Governorate_Clean'] = df[col_governorate].apply(extract_governorate_name)
    col_governorate = 'Governorate_Clean'

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

# Secondary area filter (if available)
area_choice = None
if col_area:
    st.sidebar.markdown("### 🏘️ Sub-Area Filter")
    
    # Filter areas based on selected governorates
    if governorate_choice and len(governorate_choice) > 0:
        available_areas = df[df[col_governorate].isin(governorate_choice)][col_area].dropna().unique()
    else:
        available_areas = df[col_area].dropna().unique()
    
    available_areas = sorted(available_areas.tolist())
    
    if available_areas:
        # Quick selection for areas
        col_all_area, col_none_area = st.sidebar.columns(2)
        with col_all_area:
            if st.button("✅ All Areas", key="select_all_areas", help="Select all sub-areas"):
                st.session_state.area_choice = available_areas
                st.rerun()
        with col_none_area:
            if st.button("❌ No Areas", key="deselect_all_areas", help="Deselect all sub-areas"):
                st.session_state.area_choice = []
                st.rerun()
        
        # Filter session state areas to only include available ones
        valid_area_choice = [area for area in st.session_state.area_choice if area in available_areas]
        if not valid_area_choice and available_areas:
            valid_area_choice = available_areas
        
        area_choice = st.sidebar.multiselect(
            f"🏘️ Select {col_area}s",
            options=available_areas,
            default=valid_area_choice,
            help="Further filter by specific areas within selected regions"
        )
        
        st.session_state.area_choice = area_choice
        
        # Show area selection summary
        if len(area_choice) != len(available_areas):
            area_coverage = (len(area_choice) / len(available_areas)) * 100 if available_areas else 0
            st.sidebar.info(f"🏘️ Areas: {len(area_choice)}/{len(available_areas)} ({area_coverage:.1f}%)")

# Geographic Analysis Mode
st.sidebar.markdown("### 🔍 Geographic Analysis Mode")
geo_analysis_mode = st.sidebar.selectbox(
    "Analysis Focus:",
    ["Standard Analysis", "Compare Regions", "Regional Ranking", "Geographic Distribution"],
    help="Choose how to analyze the geographic data"
)

# Sidebar controls for other filters
st.sidebar.markdown("## 🎛️ Other Interactive Controls")

# Initiative filter
selected_initiatives = None
if col_initiative:
    uniq_init = sorted(df[col_initiative].dropna().unique().tolist())
    selected_initiatives = st.sidebar.multiselect(
        "🏗️ Initiative Status Filter",
        options=uniq_init,
        default=uniq_init,
        help="Filter by existence of tourism initiatives"
    )
else:
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_cols:
        selected_cat = st.sidebar.selectbox("📊 Choose categorical column", options=[None] + categorical_cols)
        if selected_cat:
            col_initiative = selected_cat
            uniq_init = sorted(df[col_initiative].dropna().unique().tolist())
            selected_initiatives = st.sidebar.multiselect(f"Filter by {selected_cat}", options=uniq_init, default=uniq_init)

# Metric and aggregation selection
if preferred_metrics:
    metric = st.sidebar.selectbox("📈 Select Metric to Analyze", preferred_metrics, 
                                help="Choose the numeric variable for analysis")
    agg_func = st.sidebar.selectbox("🔢 Aggregation Method", 
                                  ["mean", "median", "sum", "count"], 
                                  index=0,
                                  help="How to aggregate the metric by groups")
else:
    st.error("❌ No numeric columns found in the dataset!")
    st.stop()

# Apply filters
df_filtered = df.copy()
filter_steps = []

# Apply governorate filter
if governorate_choice is not None and len(governorate_choice) > 0:
    df_filtered = df_filtered[df_filtered[col_governorate].isin(governorate_choice)]
    filter_steps.append(f"Governorate: {len(governorate_choice)} selected")

# Apply area filter
if area_choice is not None and len(area_choice) > 0 and col_area:
    df_filtered = df_filtered[df_filtered[col_area].isin(area_choice)]
    filter_steps.append(f"Areas: {len(area_choice)} selected")

# Apply initiative filter
if col_initiative and selected_initiatives is not None and len(selected_initiatives) > 0:
    df_filtered = df_filtered[df_filtered[col_initiative].isin(selected_initiatives)]
    filter_steps.append(f"Initiatives: {len(selected_initiatives)} selected")

# Display filter summary
if len(df_filtered) != len(df):
    st.info(f"📊 Showing {len(df_filtered):,} out of {len(df):,} records | Filters: {' | '.join(filter_steps)}")

# Enhanced key metrics display
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_records = len(df_filtered)
    st.metric("📊 Records", f"{total_records:,}")

with col2:
    if col_governorate and governorate_choice:
        regions_count = len(governorate_choice)
        st.metric("🏛️ Regions", regions_count)

with col3:
    if col_area and area_choice:
        areas_count = len(area_choice)
        st.metric("🏘️ Areas", areas_count)

with col4:
    if metric in df_filtered.columns and len(df_filtered) > 0:
        avg_metric = df_filtered[metric].mean()
        st.metric(f"📈 Avg {metric.split()[-1] if len(metric.split()) > 1 else metric}", f"{avg_metric:.1f}" if not pd.isna(avg_metric) else "N/A")

with col5:
    if selected_initiatives:
        init_count = len(selected_initiatives)
        st.metric("🏗️ Initiatives", init_count)

# ENHANCED GEOGRAPHIC ANALYSIS SECTION
if col_governorate and governorate_choice and geo_analysis_mode != "Standard Analysis":
    st.markdown('<div class="sub-header">🗺️ Geographic Analysis Results</div>', unsafe_allow_html=True)
    
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
            ranking_df = ranking_df[['Rank', col_governorate, 'Average', 'Total', 'Count',
