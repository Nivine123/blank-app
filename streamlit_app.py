import streamlit as st
import pandas as pd
import plotly.express as px

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

# Column names for filtering and analysis
column_initiatives = 'Existence of initiatives and projects in the past five years to improve the tourism sector - exists'
column_tourism_index = 'Tourism Index'
column_ref_area = 'refArea'

# Load data (either from URL or file upload)
df = None
err = None
use_url = st.sidebar.checkbox("Load dataset from URL", value=True)

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

# Data preview
st.sidebar.markdown("## 📊 Dataset Information")
st.sidebar.metric("Total Rows", f"{df.shape[0]:,}")
st.sidebar.metric("Total Columns", f"{df.shape[1]}")

# Expandable data preview
with st.expander("🔍 Preview Dataset (First 5 Rows)"):
    st.dataframe(df.head(), use_container_width=True)

# Regional and Initiative filtering
df_initiatives = df[df[column_initiatives] == 1]

# Count initiatives by region
initiative_counts_by_region = df_initiatives[column_ref_area].value_counts().reset_index()
initiative_counts_by_region.columns = ['Region', 'Number of Initiatives']

# Calculate average tourism index by region
tourism_index_by_region = df_initiatives.groupby(column_ref_area)[column_tourism_index].mean().reset_index()
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

# Export functionality
if st.button("📥 Export Filtered Data"):
    csv = df_initiatives.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_tourism_data.csv",
        mime="text/csv"
    )

# Add region-based analysis to the original dashboard
st.markdown('<div class="sub-header">🗺️ Geographic Insights: Initiatives by Region</div>', unsafe_allow_html=True)

# Add region-based analysis
region_analysis = df.groupby(column_ref_area)[column_initiatives].sum().reset_index()
region_analysis.columns = ['Region', 'Initiatives Count']

fig_region = px.bar(
    region_analysis,
    x='Region',
    y='Initiatives Count',
    title='Total Initiatives by Region',
    labels={'Region': 'Region', 'Initiatives Count': 'Number of Initiatives'},
    color='Initiatives Count',
    color_continuous_scale='Viridis',
    hover_data=['Initiatives Count']
)

fig_region.update_layout(
    xaxis={'tickangle': 45},  # Rotate region names
    showlegend=False
)

# Show the region-based initiatives chart
st.plotly_chart(fig_region)

# Additional insights can be added below as required
