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
col_ref_area = find_col(df, ["refArea", "ref_area", "reference_area"])

# Look for additional geographic columns
col_area = find_col(df, ["Area", "City", "Municipality", "District", "Caza", "area", "city"])

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

# ENHANCED REGIONAL FILTERING SECTION
st.sidebar.markdown("## 🗺️ Enhanced Geographic Filters")

# Primary region filter
region_choice = None
region_col = col_governorate or col_ref_area

if region_col:
    st.sidebar.markdown("### 🏛️ Primary Region Filter")
    
    uniq_regions = sorted(df[region_col].dropna().unique().tolist())
    
    # Quick selection buttons
    col_all, col_none = st.sidebar.columns(2)
    with col_all:
        select_all = st.button("✅ All", key="select_all_regions")
    with col_none:
        select_none = st.button("❌ None", key="deselect_all_regions")
    
    # Initialize default selection
    if 'region_choice' not in st.session_state:
        st.session_state.region_choice = uniq_regions
    
    if select_all:
        st.session_state.region_choice = uniq_regions
    elif select_none:
        st.session_state.region_choice = []
    
    # Main region selector
    region_choice = st.sidebar.multiselect(
        f"🏛️ Select Regions",
        options=uniq_regions,
        default=st.session_state.region_choice,
        help="Filter analysis by specific regions",
        key="region_multiselect"
    )
    
    st.session_state.region_choice = region_choice
    
    # Show selection summary
    if len(region_choice) != len(uniq_regions):
        coverage_pct = (len(region_choice) / len(uniq_regions)) * 100
        st.sidebar.info(f"📊 Selected: {len(region_choice)}/{len(uniq_regions)} regions ({coverage_pct:.1f}%)")

# NEW: Interactive controls for regional initiatives visualization
st.sidebar.markdown("## 🎯 Initiatives Visualization Controls")

# Interactive Feature 1: Chart Type Selection
initiatives_chart_type = st.sidebar.selectbox(
    "📊 Initiatives Chart Type:",
    ["Bar Chart", "Horizontal Bar Chart", "Pie Chart", "Donut Chart"],
    help="Choose how to display regional initiatives data"
)

# Interactive Feature 2: Tourism Index Integration
show_tourism_index = st.sidebar.checkbox(
    "📈 Include Tourism Index in Regional Analysis",
    value=True,
    help="Add tourism index as color coding or secondary information"
)

# Sorting options for initiatives
sort_initiatives = st.sidebar.selectbox(
    "🔄 Sort Initiatives By:",
    ["Count (Descending)", "Count (Ascending)", "Alphabetical", "Tourism Index"],
    help="Choose how to sort regions in the initiatives visualization"
)

# Secondary area filter (if available)
area_choice = None
if col_area:
    st.sidebar.markdown("### 🏘️ Sub-Area Filter")
    
    # Filter areas based on selected regions
    if region_choice and len(region_choice) > 0:
        available_areas = df[df[region_col].isin(region_choice)][col_area].dropna().unique()
    else:
        available_areas = df[col_area].dropna().unique()
    
    available_areas = sorted(available_areas.tolist())
    
    if available_areas:
        area_choice = st.sidebar.multiselect(
            f"🏘️ Select Areas",
            options=available_areas,
            default=available_areas,
            help="Further filter by specific areas within selected regions"
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

# Metric selection
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

# Apply region filter
if region_choice is not None and len(region_choice) > 0:
    df_filtered = df_filtered[df_filtered[region_col].isin(region_choice)]
    filter_steps.append(f"Regions: {len(region_choice)} selected")

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
    if region_col and region_choice:
        regions_count = len(region_choice)
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
    if col_initiative and selected_initiatives:
        # Count records with initiatives = 1
        initiative_count = len(df_filtered[df_filtered[col_initiative] == 1])
        st.metric("🏗️ Active Initiatives", initiative_count)

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
    if agg_func == "mean":
        agg_df = df_filtered.groupby(col_initiative)[metric].mean().reset_index()
    elif agg_func == "median":
        agg_df = df_filtered.groupby(col_initiative)[metric].median().reset_index()
    elif agg_func == "sum":
        agg_df = df_filtered.groupby(col_initiative)[metric].sum().reset_index()
    else:  # count
        agg_df = df_filtered.groupby(col_initiative)[metric].count().reset_index()
    
    agg_df = agg_df.sort_values(by=metric, ascending=False)
    
    # Create enhanced bar chart
    fig1 = px.bar(
        agg_df, 
        x=col_initiative, 
        y=metric,
        title=f"{agg_func.title()} of {metric} by {col_initiative}",
        labels={col_initiative: col_initiative, metric: f"{agg_func.title()} {metric}"},
        color=metric,
        color_continuous_scale="viridis",
        text=metric
    )
    
    # Customize the chart
    fig1.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig1.update_layout(
        height=500,
        xaxis={'categoryorder': 'total descending'},
        showlegend=False
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Enhanced insights
    insights_text = f"""
    <div class="insight-box">
    <strong>💡 Key Insights:</strong><br>
    • Highest {agg_func}: <strong>{agg_df.iloc[0][col_initiative]}</strong> with {agg_df.iloc[0][metric]:.2f}<br>
    • Lowest {agg_func}: <strong>{agg_df.iloc[-1][col_initiative]}</strong> with {agg_df.iloc[-1][metric]:.2f}<br>
    • Range: {agg_df[metric].max() - agg_df[metric].min():.2f} ({metric})
    </div>
    """
    st.markdown(insights_text, unsafe_allow_html=True)

# NEW: Visualization 2 - Tourism Initiatives by Region
st.markdown('<div class="sub-header">🗺️ Visualization 2: Tourism Initiatives by Region</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This visualization shows the number of tourism initiatives across different regions.
The interactive features allow you to explore the data from multiple perspectives and understand the relationship 
between initiative activity and tourism performance.
</div>
""", unsafe_allow_html=True)

if region_col and col_initiative:
    # Filter the data to include only rows where initiatives exist (value = 1)
    initiatives_exist_df = df_filtered[df_filtered[col_initiative] == 1]
    
    if len(initiatives_exist_df) > 0:
        # Count the number of initiatives per region
        initiative_counts_by_region = initiatives_exist_df[region_col].value_counts().reset_index()
        initiative_counts_by_region.columns = [region_col, 'Number of Initiatives']
        
        # Add tourism index if available and selected
        regional_data = initiative_counts_by_region.copy()
        
        if show_tourism_index and col_tourism_index in df.columns:
            # Calculate average tourism index by region for initiatives
            tourism_index_by_region = initiatives_exist_df.groupby(region_col)[col_tourism_index].mean().reset_index()
            tourism_index_by_region.columns = [region_col, 'Average Tourism Index']
            
            # Merge the dataframes
            regional_data = pd.merge(initiative_counts_by_region, tourism_index_by_region, on=region_col)
        
        # Apply sorting based on user selection
        if sort_initiatives == "Count (Descending)":
            regional_data = regional_data.sort_values('Number of Initiatives', ascending=False)
        elif sort_initiatives == "Count (Ascending)":
            regional_data = regional_data.sort_values('Number of Initiatives', ascending=True)
        elif sort_initiatives == "Alphabetical":
            regional_data = regional_data.sort_values(region_col, ascending=True)
        elif sort_initiatives == "Tourism Index" and 'Average Tourism Index' in regional_data.columns:
            regional_data = regional_data.sort_values('Average Tourism Index', ascending=False)
        
        # Create visualization based on interactive controls
        if initiatives_chart_type == "Bar Chart":
            if show_tourism_index and 'Average Tourism Index' in regional_data.columns:
                fig2 = px.bar(
                    regional_data,
                    x=region_col,
                    y='Number of Initiatives',
                    title='Number of Tourism Initiatives by Region',
                    color='Average Tourism Index',
                    color_continuous_scale='Viridis',
                    hover_data=['Average Tourism Index'],
                    text='Number of Initiatives',
                    labels={region_col: 'Region'}
                )
            else:
                fig2 = px.bar(
                    regional_data,
                    x=region_col,
                    y='Number of Initiatives',
                    title='Number of Tourism Initiatives by Region',
                    color='Number of Initiatives',
                    color_continuous_scale='Blues',
                    text='Number of Initiatives',
                    labels={region_col: 'Region'}
                )
            
            fig2.update_traces(texttemplate='%{text}', textposition='outside')
            fig2.update_layout(xaxis_tickangle=45, height=500)
            
        elif initiatives_chart_type == "Horizontal Bar Chart":
            if show_tourism_index and 'Average Tourism Index' in regional_data.columns:
                fig2 = px.bar(
                    regional_data,
                    x='Number of Initiatives',
                    y=region_col,
                    orientation='h',
                    title='Number of Tourism Initiatives by Region (Horizontal)',
                    color='Average Tourism Index',
                    color_continuous_scale='Viridis',
                    hover_data=['Average Tourism Index'],
                    text='Number of Initiatives',
                    labels={region_col: 'Region'}
                )
            else:
                fig2 = px.bar(
                    regional_data,
                    x='Number of Initiatives',
                    y=region_col,
                    orientation='h',
                    title='Number of Tourism Initiatives by Region (Horizontal)',
                    color='Number of Initiatives',
                    color_continuous_scale='Blues',
                    text='Number of Initiatives',
                    labels={region_col: 'Region'}
                )
            
            fig2.update_traces(texttemplate='%{text}', textposition='outside')
            fig2.update_layout(height=max(400, len(regional_data) * 30))
            
        elif initiatives_chart_type == "Pie Chart":
            fig2 = px.pie(
                regional_data,
                values='Number of Initiatives',
                names=region_col,
                title='Distribution of Tourism Initiatives by Region'
            )
            fig2.update_layout(height=500)
            
        else:  # Donut Chart
            fig2 = px.pie(
                regional_data,
                values='Number of Initiatives',
                names=region_col,
                title='Distribution of Tourism Initiatives by Region (Donut)',
                hole=0.4
            )
            fig2.update_layout(height=500)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Generate insights based on the visualization
        total_initiatives = regional_data['Number of Initiatives'].sum()
        most_active_region = regional_data.iloc[0][region_col]
        max_initiatives = regional_data.iloc[0]['Number of Initiatives']
        avg_initiatives = regional_data['Number of Initiatives'].mean()
        
        # Display insights in columns
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
            if show_tourism_index and 'Average Tourism Index' in regional_data.columns:
                avg_tourism_idx = regional_data['Average Tourism Index'].mean()
                best_tourism_region = regional_data.loc[regional_data['Average Tourism Index'].idxmax(), region_col]
                st.markdown(f"""
                <div class="context-box">
                <strong>📈 Tourism Performance:</strong><br>
                Best Index: {best_tourism_region}<br>
                <strong>Avg Index:</strong> {avg_tourism_idx:.2f}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="context-box">
                <strong>📈 Initiative Distribution:</strong><br>
                Avg per Region: {avg_initiatives:.1f}<br>
                <strong>Range:</strong> {regional_data['Number of Initiatives'].min()}-{regional_data['Number of Initiatives'].max()}
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed regional data table
        st.markdown("#### 📋 Detailed Regional Initiative Data")
        st.dataframe(regional_data, use_container_width=True, hide_index=True)
        
        # Additional correlation analysis if tourism index is included
        if show_tourism_index and 'Average Tourism Index' in regional_data.columns and len(regional_data) > 2:
            correlation = regional_data['Number of Initiatives'].corr(regional_data['Average Tourism Index'])
            
            st.markdown("#### 🔍 Initiative-Performance Relationship")
            
            if correlation > 0.5:
                correlation_insight = "Strong positive correlation - regions with more initiatives tend to have higher tourism performance"
                correlation_color = "geographic-box"
            elif correlation > 0.3:
                correlation_insight = "Moderate positive correlation - some relationship between initiatives and tourism performance"
                correlation_color = "context-box"
            elif correlation < -0.3:
                correlation_insight = "Negative correlation - regions with more initiatives may have lower current tourism performance (potential development areas)"
                correlation_color = "insight-box"
            else:
                correlation_insight = "Low correlation - initiative activity and tourism performance appear independent"
                correlation_color = "context-box"
            
            st.markdown(f"""
            <div class="{correlation_color}">
            <strong>📈 Initiative-Performance Relationship:</strong><br>
            Correlation coefficient: {correlation:.3f}<br>
            <strong>Interpretation:</strong> {correlation_insight}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ No regions with active initiatives found in the filtered data.")

else:
    st.error("❌ Cannot create initiatives visualization - missing required columns for regional or initiative data.")

# Additional Analysis Section
st.markdown('<div class="sub-header">🔍 Additional Interactive Analysis</div>', 
            unsafe_allow_html=True)

analysis_type = st.selectbox(
    "Choose additional analysis:",
    ["Summary Statistics", "Correlation Analysis", "Top Performers", "Geographic Insights"]
)

if analysis_type == "Summary Statistics":
    st.markdown("### 📊 Comprehensive Statistics")
    numeric_summary = df_filtered[numeric_cols].describe()
    st.dataframe(numeric_summary, use_container_width=True)

elif analysis_type == "Correlation Analysis" and len(numeric_cols) > 1:
    st.markdown("### 🔗 Correlation Matrix")
    corr_matrix = df_filtered[numeric_cols].corr()
    fig_corr = px.imshow(
        corr_matrix,
        aspect='auto',
        color_continuous_scale='RdBu',
        title='Correlation Matrix of Numeric Variables'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_type == "Top Performers":
    st.markdown("### 🏆 Top Performers")
    if region_col and metric and len(df_filtered) > 0:
        top_regions = df_filtered.groupby(region_col)[metric].mean().nlargest(5)
        col_a, col_b = st.columns(2)
        with col_a:
            st.bar_chart(top_regions)
        with col_b:
            st.write("**Top 5 Regions:**")
            for region, value in top_regions.items():
                st.write(f"• {region}: {value:.2f}")

elif analysis_type == "Geographic Insights":
    st.markdown("### 🗺️ Geographic Performance Insights")
    
    if region_col and len(df_filtered) > 0:
        geo_insights = df_filtered.groupby(region_col)[metric].agg(['mean', 'count', 'std']).round(2)
        geo_insights.columns = ['Average', 'Data Points', 'Variability']
        geo_insights = geo_insights.sort_values('Average', ascending=False)
        
        # Calculate regional statistics
        top_region = geo_insights.index[0]
        top_value = geo_insights.iloc[0]['Average']
        bottom_region = geo_insights.index[-1]
        bottom_value = geo_insights.iloc[-1]['Average']
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown(f"""
            <div class="geographic-box">
            <strong>🏆 Best Performer:</strong> {top_region}<br>
            <strong>Score:</strong> {top_value:.2f}<br>
            <strong>🔻 Needs Improvement:</strong> {bottom_region}<br>
            <strong>Score:</strong> {bottom_value:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        with col_insight2:
            performance_gap = top_value - bottom_value
            avg_variability = geo_insights['Variability'].mean()
            st.markdown(f"""
            <div class="geographic-box">
            <strong>📊 Performance Gap:</strong> {performance_gap:.2f}<br>
            <strong>📈 Avg Variability:</strong> {avg_variability:.2f}<br>
            <strong>🗺️ Regions Analyzed:</strong> {len(geo_insights)}
            </div>
            """, unsafe_allow_html=True)
        
        st.dataframe(geo_insights, use_container_width=True)

# Footer with usage instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Use This Enhanced Dashboard:

1. **Geographic Filtering**: 
   - Use quick selection buttons (All/None) for efficient region management
   - Multi-level filtering available when area data is present

2. **Interactive Features**:
   - **Feature 1**: Dynamic chart type selection for initiatives visualization (Bar, Horizontal Bar, Pie, Donut)
   - **Feature 2**: Tourism index integration toggle to add performance context to initiative analysis

3. **Data Analysis**: 
   - Switch between different tourism metrics and aggregation methods
   - Sort initiatives data by count, alphabetical order, or tourism performance
   - Explore correlation between initiative activity and tourism performance

### 💡 Key Design Decisions:

- **Flexible Column Detection**: Automatically finds relevant columns regardless of exact naming
- **Interactive Chart Controls**: Multiple visualization types for the initiatives data
- **Performance Context**: Optional tourism index integration for deeper insights
- **Smart Sorting**: Multiple sorting options to reveal different data patterns
- **Correlation Analysis**: Automatic relationship detection between initiatives and performance
""")

# Export functionality
if st.button("📥 Export Filtered Data"):
    csv = df_filtered.to_csv(index=False)
    
    # Create filename with filter info
    filter_info = []
    if region_choice and len(region_choice) < len(df[region_col].unique()):
        filter_info.append(f"{len(region_choice)}regions")
    
    filter_suffix = "_" + "_".join(filter_info) if filter_info else ""
    
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"filtered_tourism_data{filter_suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
