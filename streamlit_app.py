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
</div>
""", unsafe_allow_html=True)

# Dataset loading and reading
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

# Interactive Feature 1: Chart Type Selection
st.sidebar.markdown("## 🎯 Regional Visualization Controls")
chart_type = st.sidebar.selectbox(
    "📊 Chart Type for Regional Analysis:",
    ["Bar Chart", "Horizontal Bar Chart", "Pie Chart"],
    help="Choose how to display the regional initiatives data"
)

# Interactive Feature 2: Show/Hide Tourism Index
show_tourism_index = st.sidebar.checkbox(
    "📈 Include Tourism Index Colors",
    value=True,
    help="Add tourism index as color coding"
)

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

# Visualization 1: Original Bar Chart from your second code
st.markdown('<div class="sub-header">📊 Visualization 1: Number of Tourism Initiatives and Average Tourism Index by Region</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This visualization shows the number of tourism initiatives and average tourism index by region.
It combines both initiative activity and tourism performance in one comprehensive view.
</div>
""", unsafe_allow_html=True)

# Count initiatives by region
initiative_counts_by_region = df_initiatives[column_ref_area].value_counts().reset_index()
initiative_counts_by_region.columns = ['Region', 'Number of Initiatives']

# Calculate average tourism index by region
tourism_index_by_region = df_initiatives.groupby(column_ref_area)[column_tourism_index].mean().reset_index()
tourism_index_by_region.columns = ['Region', 'Average Tourism Index']

# Merge the two dataframes
merged_df = pd.merge(initiative_counts_by_region, tourism_index_by_region, on='Region')

# Bar chart: Number of initiatives and average tourism index by region
fig1 = px.bar(
    merged_df,
    x='Region',
    y='Number of Initiatives',
    title='Number of Tourism Initiatives and Average Tourism Index by Region',
    labels={'Region': 'Region', 'Number of Initiatives': 'Number of Initiatives'},
    color='Average Tourism Index',
    color_continuous_scale='Viridis',
    hover_data=['Average Tourism Index'],
    text='Number of Initiatives'
)

# Update layout for better readability
fig1.update_layout(
    barmode='group',
    xaxis={'tickangle': 45},  # Rotate region names
    showlegend=False,
    height=500
)

fig1.update_traces(texttemplate='%{text}', textposition='outside')

# Show the plot
st.plotly_chart(fig1, use_container_width=True)

# Insights for Visualization 1
top_region_initiatives = merged_df.loc[merged_df['Number of Initiatives'].idxmax()]
top_region_tourism = merged_df.loc[merged_df['Average Tourism Index'].idxmax()]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="geographic-box">
    <strong>🏆 Most Initiatives:</strong><br>
    {top_region_initiatives['Region']}<br>
    <strong>Count:</strong> {top_region_initiatives['Number of Initiatives']}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="insight-box">
    <strong>📈 Highest Tourism Index:</strong><br>
    {top_region_tourism['Region']}<br>
    <strong>Index:</strong> {top_region_tourism['Average Tourism Index']:.2f}
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_initiatives = merged_df['Number of Initiatives'].sum()
    avg_tourism_index = merged_df['Average Tourism Index'].mean()
    st.markdown(f"""
    <div class="context-box">
    <strong>📊 Overall Stats:</strong><br>
    Total Initiatives: {total_initiatives}<br>
    <strong>Avg Tourism Index:</strong> {avg_tourism_index:.2f}
    </div>
    """, unsafe_allow_html=True)

# NEW Visualization 2: Interactive Regional Initiatives Analysis
st.markdown('<div class="sub-header">🗺️ Visualization 2: Interactive Regional Initiatives Analysis</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This interactive visualization focuses specifically on tourism initiatives by region.
You can change the chart type and toggle tourism index coloring to explore the data from different perspectives.
</div>
""", unsafe_allow_html=True)

# Create visualization based on selected chart type
if chart_type == "Bar Chart":
    if show_tourism_index:
        fig2 = px.bar(
            merged_df.sort_values('Number of Initiatives', ascending=False),
            x='Region',
            y='Number of Initiatives',
            title='Tourism Initiatives by Region (Interactive)',
            color='Average Tourism Index',
            color_continuous_scale='plasma',
            hover_data=['Average Tourism Index'],
            text='Number of Initiatives'
        )
    else:
        fig2 = px.bar(
            merged_df.sort_values('Number of Initiatives', ascending=False),
            x='Region',
            y='Number of Initiatives',
            title='Tourism Initiatives by Region (Interactive)',
            color='Number of Initiatives',
            color_continuous_scale='blues',
            text='Number of Initiatives'
        )
    
    fig2.update_traces(texttemplate='%{text}', textposition='outside')
    fig2.update_layout(xaxis_tickangle=45)
    
elif chart_type == "Horizontal Bar Chart":
    if show_tourism_index:
        fig2 = px.bar(
            merged_df.sort_values('Number of Initiatives', ascending=True),
            x='Number of Initiatives',
            y='Region',
            orientation='h',
            title='Tourism Initiatives by Region (Horizontal)',
            color='Average Tourism Index',
            color_continuous_scale='plasma',
            hover_data=['Average Tourism Index'],
            text='Number of Initiatives'
        )
    else:
        fig2 = px.bar(
            merged_df.sort_values('Number of Initiatives', ascending=True),
            x='Number of Initiatives',
            y='Region',
            orientation='h',
            title='Tourism Initiatives by Region (Horizontal)',
            color='Number of Initiatives',
            color_continuous_scale='blues',
            text='Number of Initiatives'
        )
    
    fig2.update_traces(texttemplate='%{text}', textposition='outside')
    
else:  # Pie Chart
    fig2 = px.pie(
        merged_df,
        values='Number of Initiatives',
        names='Region',
        title='Distribution of Tourism Initiatives by Region',
        hover_data=['Average Tourism Index'] if show_tourism_index else None
    )

fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# Interactive insights for Visualization 2
st.markdown("#### 📋 Regional Performance Analysis")

# Create performance categories
merged_df['Performance_Category'] = pd.cut(
    merged_df['Average Tourism Index'], 
    bins=3, 
    labels=['🔴 Needs Improvement', '🟡 Average', '🟢 High Performing']
)

# Show categorized results
performance_summary = merged_df.groupby('Performance_Category').agg({
    'Number of Initiatives': ['count', 'sum', 'mean'],
    'Average Tourism Index': 'mean'
}).round(2)

st.dataframe(performance_summary, use_container_width=True)

# Detailed regional table
st.markdown("#### 📊 Complete Regional Data")
display_df = merged_df[['Region', 'Number of Initiatives', 'Average Tourism Index', 'Performance_Category']].sort_values('Number of Initiatives', ascending=False)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Interactive Features Summary
st.markdown("---")
st.markdown("""
### 🚀 Interactive Features Summary

**🎛️ Feature 1: Chart Type Selection**
- **Location**: Sidebar → "Chart Type for Regional Analysis"
- **Impact**: Changes the visualization type in Visualization 2
- **Options**: Bar Chart, Horizontal Bar Chart, Pie Chart
- **Insight**: Different chart types reveal different patterns in the same data

**🎨 Feature 2: Tourism Index Color Toggle**  
- **Location**: Sidebar → "Include Tourism Index Colors"
- **Impact**: Toggles color coding based on tourism index values
- **Value**: When enabled, shows regions with high initiatives AND high performance
- **Business Use**: Identify regions with strong tourism potential vs. high activity

### 💡 How to Generate Insights:

1. **Chart Comparison**:
   - **Bar Chart**: Easy to compare initiative counts
   - **Horizontal Bar**: Better for long region names
   - **Pie Chart**: Shows proportional distribution

2. **Color Analysis**:
   - **Tourism Index ON**: Find regions with both high activity and performance
   - **Tourism Index OFF**: Focus purely on initiative distribution
   - **Look for mismatches**: High initiatives but low tourism index (improvement opportunities)

### 🎯 For Your 1-Minute Video:
- **0-15s**: Show Chart Type switching (Bar → Horizontal → Pie)
- **15-30s**: Toggle Tourism Index colors and explain the color changes
- **30-45s**: Highlight insights (top performers, improvement opportunities)
- **45-60s**: Explain design decisions (simple interactions, clear business value)
""")

# Export functionality
if st.button("📥 Export Filtered Data"):
    csv = df_initiatives.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_tourism_data.csv",
        mime="text/csv"
    )
