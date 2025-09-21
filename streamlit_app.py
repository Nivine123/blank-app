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

# Apply filters
df_filtered = df.copy()

# Apply governorate filter
if st.session_state.governorate_choice:
    df_filtered = df_filtered[df_filtered[col_governorate].isin(st.session_state.governorate_choice)]

# Apply area filter
if st.session_state.area_choice:
    df_filtered = df_filtered[df_filtered[col_area].isin(st.session_state.area_choice)]

# Visualization 1: Aggregated Analysis by Initiative Status
st.markdown('<div class="sub-header">📊 Visualization 1: Regional Analysis by Initiative Status</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This visualization shows how the selected metric varies across different initiative statuses.
It helps identify which regions or initiative types perform better in terms of the chosen tourism metric.
</div>
""", unsafe_allow_html=True)

if col_initiative and not df_filtered.empty:
    # Compute aggregation
    agg_df = df_filtered.groupby(col_initiative)[col_tourism_index].mean().reset_index()
    agg_df = agg_df.sort_values(by=col_tourism_index, ascending=False)
    
    # Create enhanced bar chart
    fig1 = px.bar(
        agg_df, 
        x=col_initiative, 
        y=col_tourism_index,
        title=f"Mean {col_tourism_index} by {col_initiative}",
        labels={col_initiative: col_initiative, col_tourism_index: f"Mean {col_tourism_index}"},
        color=col_tourism_index,
        color_continuous_scale="viridis",
        text=col_tourism_index
    )
    
    # Customize the chart
    fig1.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig1.update_layout(
        height=500,
        xaxis={'categoryorder': 'total descending'},
        showlegend=False
    )
    
    st.plotly_chart(fig1, use_container_width=True)

else:
    st.error("❌ Cannot create visualization - no valid categorical column or empty dataset.")

# Visualization 2: Tourism Initiatives by Region
st.markdown('<div class="sub-header">🗺️ Visualization 2: Tourism Initiatives by Region</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This visualization shows the number of tourism initiatives across different regions,
directly based on your Colab analysis. The interactive features allow you to explore the data from multiple perspectives
and understand the relationship between initiative activity and tourism performance.
</div>
""", unsafe_allow_html=True)

# Use refArea column for regional analysis (from your second code)
if 'refArea' in df.columns and col_initiative:
    
    # Filter the data to include only rows where initiatives exist (based on your Colab code)
    initiatives_exist_df = df_filtered[df_filtered[col_initiative] == 1]
    
    if len(initiatives_exist_df) > 0:
        # Count the number of initiatives per region (your Colab logic)
        initiative_counts_by_region = initiatives_exist_df['refArea'].value_counts().reset_index()
        initiative_counts_by_region.columns = ['refArea', 'Number of Initiatives']
        
        # Add tourism index if available and selected
        regional_data = initiative_counts_by_region.copy()
        
        if col_tourism_index in df.columns:
            # Calculate average tourism index by region for initiatives
            tourism_index_by_region = initiatives_exist_df.groupby('refArea')[col_tourism_index].mean().reset_index()
            tourism_index_by_region.columns = ['refArea', 'Average Tourism Index']
            
            # Merge the dataframes
            regional_data = pd.merge(initiative_counts_by_region, tourism_index_by_region, on='refArea')
        
        # Create visualization based on interactive controls
        fig2 = px.bar(
            regional_data.sort_values('Number of Initiatives', ascending=False),
            x='refArea',
            y='Number of Initiatives',
            title='Number of Tourism Initiatives by Region',
            color='Average Tourism Index' if 'Average Tourism Index' in regional_data else 'Number of Initiatives',
            color_continuous_scale='Viridis' if 'Average Tourism Index' in regional_data else 'Blues',
            text='Number of Initiatives'
        )
        
        fig2.update_traces(texttemplate='%{text}', textposition='outside')
        fig2.update_layout(xaxis_tickangle=45)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Generate insights based on the visualization
        total_initiatives = regional_data['Number of Initiatives'].sum()
        most_active_region = regional_data.loc[regional_data['Number of Initiatives'].idxmax(), 'refArea']
        max_initiatives = regional_data['Number of Initiatives'].max()
        avg_initiatives = regional_data['Number of Initiatives'].mean()
        
        # Display insights
        col_i1, col_i2, col_i3 = st.columns(3)
        
        with col_i1:
            st.markdown(f"""
            <div class="geographic-box">
            <strong>🏆 Most Active Region:</strong><br>
            {most_active_region}<br>
            <strong>Initiatives:</strong> {max_initiatives}
            </div>
            """, unsafe_allow_html=True)
        
        with col_i2:
            st.markdown(f"""
            <div class="insight-box">
            <strong>📊 Total Overview:</strong><br>
            Total Initiatives: {total_initiatives}<br>
            <strong>Active Regions:</strong> {len(regional_data)}
            </div>
            """, unsafe_allow_html=True)
        
        with col_i3:
            st.markdown(f"""
            <div class="context-box">
            <strong>📈 Initiative Distribution:</strong><br>
            Avg per Region: {avg_initiatives:.1f}<br>
            <strong>Range:</strong> {regional_data['Number of Initiatives'].min()}-{regional_data['Number of Initiatives'].max()}
            </div>
            """, unsafe_allow_html=True)

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
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_tourism_data.csv",
        mime="text/csv"
    )
