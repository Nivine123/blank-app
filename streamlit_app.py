import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
.performance-box {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
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
The visualizations help identify patterns in tourism development, infrastructure distribution, and regional performance.
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

@st.cache_data
def create_regional_performance_matrix(df_filtered, selected_metrics, col_governorate):
    """Create comprehensive regional performance matrix"""
    if df_filtered.empty or not selected_metrics:
        return pd.DataFrame()
    
    regional_matrix = {}
    for metric in selected_metrics:
        if metric in df_filtered.columns:
            regional_stats = df_filtered.groupby(col_governorate)[metric].agg(['mean', 'median', 'count']).round(2)
            for stat in ['mean', 'median', 'count']:
                regional_matrix[f"{metric}_{stat}"] = regional_stats[stat]
    
    if regional_matrix:
        matrix_df = pd.DataFrame(regional_matrix)
        # Calculate composite score (average of normalized means)
        mean_cols = [col for col in matrix_df.columns if col.endswith('_mean')]
        if mean_cols:
            normalized_means = matrix_df[mean_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x)
            matrix_df['composite_score'] = normalized_means.mean(axis=1).round(2)
        return matrix_df
    return pd.DataFrame()

@st.cache_data  
def calculate_regional_clusters(performance_data):
    """Apply clustering algorithm to identify regional performance groups"""
    if len(performance_data) < 3:
        return np.zeros(len(performance_data))
    
    # Select numeric columns only
    numeric_cols = performance_data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return np.zeros(len(performance_data))
    
    try:
        # Standardize metrics
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(performance_data[numeric_cols].fillna(0))
        
        # Apply clustering
        n_clusters = min(3, len(performance_data))
        clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(standardized_data)
        return clusters
    except:
        return np.zeros(len(performance_data))

def generate_regional_insights(df_regional, metric):
    """Generate contextual insights for regional performance"""
    if df_regional.empty:
        return {}
    
    insights = {}
    if 'composite_score' in df_regional.columns:
        insights['top_performer'] = df_regional['composite_score'].idxmax()
        insights['top_score'] = df_regional['composite_score'].max()
        insights['bottom_performer'] = df_regional['composite_score'].idxmin()
        insights['bottom_score'] = df_regional['composite_score'].min()
        insights['performance_range'] = insights['top_score'] - insights['bottom_score']
        insights['average_performance'] = df_regional['composite_score'].mean()
    
    return insights

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
col_governorate = find_col(df, ["Governorate", "governorate", "Region", "region", "Mohafazat", "mohafazat", "refArea"])

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

# NEW: Regional Analysis Controls
st.sidebar.markdown("## 🗺️ Regional Analysis Controls")

# Performance benchmarking selector
benchmark_type = st.sidebar.selectbox(
    "🎯 Benchmark Analysis:",
    ["Against National Average", "Top Performers Only", "Regional Clusters", "Performance Quartiles"]
)

# Multi-metric selector for regional analysis
if preferred_metrics:
    regional_metrics = st.sidebar.multiselect(
        "📊 Select Metrics for Regional Analysis:",
        preferred_metrics,
        default=preferred_metrics[:3] if len(preferred_metrics) >= 3 else preferred_metrics,
        help="Choose 2-4 metrics for comprehensive regional comparison"
    )

# Performance threshold slider
performance_threshold = st.sidebar.slider(
    "🎚️ Performance Threshold (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    help="Set minimum performance level for regional analysis"
)

# Regional grouping options
grouping_mode = st.sidebar.selectbox(
    "🏘️ Regional Grouping:",
    ["Individual Regions", "Performance Tiers", "Statistical Clusters"]
)

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
            uniq_init = sorted(df[selected_cat].dropna().unique().tolist())
            selected_initiatives = st.sidebar.multiselect(f"Filter by {selected_cat}", options=uniq_init, default=uniq_init)

# Metric and aggregation selection
if preferred_metrics:
    metric = st.sidebar.selectbox("📈 Select Primary Metric", preferred_metrics, 
                                help="Choose the main numeric variable for analysis")
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

# Visualization 1: Aggregated Analysis (existing)
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
    
    # Enhanced insights with geographic context
    insights_text = f"""
    <div class="insight-box">
    <strong>💡 Key Insights:</strong><br>
    • Highest {agg_func}: <strong>{agg_df.iloc[0][col_initiative]}</strong> with {agg_df.iloc[0][metric]:.2f}<br>
    • Lowest {agg_func}: <strong>{agg_df.iloc[-1][col_initiative]}</strong> with {agg_df.iloc[-1][metric]:.2f}<br>
    • Range: {agg_df[metric].max() - agg_df[metric].min():.2f} ({metric})
    """
    
    if col_governorate and len(governorate_choice) > 0:
        top_region = df_filtered.groupby(col_governorate)[metric].mean().idxmax()
        insights_text += f"<br>• Top performing region: <strong>{top_region}</strong>"
    
    insights_text += "</div>"
    st.markdown(insights_text, unsafe_allow_html=True)

else:
    st.error("❌ Cannot create visualization - no valid categorical column or empty dataset.")

# NEW: Visualization 2: Comprehensive Regional Performance Analysis
st.markdown('<div class="sub-header">🗺️ Visualization 2: Comprehensive Regional Performance Analysis</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This advanced regional analysis provides multi-dimensional insights into tourism performance 
across geographic regions, enabling identification of high-performing areas and improvement opportunities through 
interactive benchmarking and clustering analysis.
</div>
""", unsafe_allow_html=True)

if col_governorate and governorate_choice and regional_metrics and len(df_filtered) > 0:
    
    # Create regional performance matrix
    regional_performance = create_regional_performance_matrix(df_filtered, regional_metrics, col_governorate)
    
    if not regional_performance.empty:
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Matrix", "🔍 Scatter Analysis", "🏆 Rankings & Clusters", "📈 Benchmarks"])
        
        with tab1:
            st.markdown("#### 📊 Regional Performance Heatmap")
            
            # Select columns for heatmap (means only)
            mean_cols = [col for col in regional_performance.columns if col.endswith('_mean')]
            
            if mean_cols:
                # Create heatmap
                fig_heatmap = px.imshow(
                    regional_performance[mean_cols].T,
                    aspect='auto',
                    color_continuous_scale='RdYlBu_r',
                    title='Regional Performance Heatmap: Tourism Indicators',
                    labels=dict(x="Regions", y="Metrics", color="Performance Score")
                )
                
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Performance insights
                insights = generate_regional_insights(regional_performance, metric)
                if insights:
                    col_i1, col_i2, col_i3 = st.columns(3)
                    with col_i1:
                        st.markdown(f"""
                        <div class="performance-box">
                        <strong>🏆 Top Performer:</strong><br>
                        {insights.get('top_performer', 'N/A')}<br>
                        <strong>Score:</strong> {insights.get('top_score', 0):.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_i2:
                        st.markdown(f"""
                        <div class="insight-box">
                        <strong>📊 Performance Range:</strong><br>
                        Span: {insights.get('performance_range', 0):.2f}<br>
                        <strong>Average:</strong> {insights.get('average_performance', 0):.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_i3:
                        st.markdown(f"""
                        <div class="geographic-box">
                        <strong>🎯 Needs Focus:</strong><br>
                        {insights.get('bottom_performer', 'N/A')}<br>
                        <strong>Score:</strong> {insights.get('bottom_score', 0):.2f}
                        </div>
                        """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("#### 🔍 Interactive Regional Scatter Analysis")
            
            # Metric selectors for scatter plot
            col_x, col_y = st.columns(2)
            with col_x:
                metric_x = st.selectbox("📈 X-Axis Metric:", regional_metrics, key="scatter_x")
            with col_y:
                metric_y = st.selectbox("📈 Y-Axis Metric:", 
                                      [m for m in regional_metrics if m != metric_x], 
                                      key="scatter_y")
            
            if metric_x and metric_y:
                # Prepare scatter data
                scatter_data = df_filtered.groupby(col_governorate).agg({
                    metric_x: 'mean',
                    metric_y: 'mean',
                    metric: 'count'  # For bubble size
                }).reset_index()
                
                scatter_data.columns = [col_governorate, f'avg_{metric_x}', f'avg_{metric_y}', 'data_points']
                
                # Add performance clusters if enough data
                if len(scatter_data) >= 3:
                    clusters = calculate_regional_clusters(scatter_data[['avg_' + metric_x, 'avg_' + metric_y]])
                    scatter_data['cluster'] = [f"Cluster {c+1}" for c in clusters]
                else:
                    scatter_data['cluster'] = 'Single Group'
                
                # Create scatter plot
                fig_scatter = px.scatter(
                    scatter_data,
                    x=f'avg_{metric_x}',
                    y=f'avg_{metric_y}',
                    size='data_points',
                    color='cluster',
                    hover_data=[col_governorate],
                    title=f'Regional Performance: {metric_x} vs {metric_y}',
                    size_max=30
                )
                
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Quadrant analysis
                x_median = scatter_data[f'avg_{metric_x}'].median()
                y_median = scatter_data[f'avg_{metric_y}'].median()
                
                st.markdown("##### 📍 Quadrant Analysis")
                high_high = scatter_data[(scatter_data[f'avg_{metric_x}'] >= x_median) & 
                                       (scatter_data[f'avg_{metric_y}'] >= y_median)]
                low_low = scatter_data[(scatter_data[f'avg_{metric_x}'] < x_median) & 
                                     (scatter_data[f'avg_{metric_y}'] < y_median)]
                
                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    st.success(f"🌟 **High Performers ({len(high_high)} regions):**")
                    for _, row in high_high.iterrows():
                        st.write(f"• {row[col_governorate]}")
                
                with col_q2:
                    st.warning(f"⚠️ **Improvement Needed ({len(low_low)} regions):**")
                    for _, row in low_low.iterrows():
                        st.write(f"• {row[col_governorate]}")
        
        with tab3:
            st.markdown("#### 🏆 Regional Rankings & Performance Clusters")
            
            # Regional rankings
            if 'composite_score' in regional_performance.columns:
                ranking_df = regional_performance.sort_values('composite_score', ascending=False).reset_index()
                ranking_df['rank'] = range(1, len(ranking_df) + 1)
                ranking_df['performance_tier'] = pd.cut(ranking_df['composite_score'], 
                                                      bins=3, 
                                                      labels=['🔴 Needs Improvement', '🟡 Average', '🟢 High Performing'])
                
                # Display rankings
                st.markdown("##### 📊 Overall Regional Rankings")
                display_cols = ['rank', col_governorate, 'composite_score', 'performance_tier']
                if all(col in ranking_df.columns for col in display_cols):
                    st.dataframe(ranking_df[display_cols].rename(columns={
                        'rank': 'Rank',
                        col_governorate: 'Region',
                        'composite_score': 'Composite Score',
                        'performance_tier': 'Performance Tier'
                    }), use_container_width=True, hide_index=True)
                
                # Performance tier distribution
                tier_counts = ranking_df['performance_tier'].value_counts()
                fig_pie = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    title="Regional Performance Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab4:
            st.markdown("#### 📈 Performance Benchmarking")
            
            if 'composite_score' in regional_performance.columns:
                # Calculate benchmarks
                national_avg = regional_performance['composite_score'].mean()
                top_quartile = regional_performance['composite_score'].quantile(0.75)
                performance_threshold_value = (regional_performance['composite_score'].max() * performance_threshold / 100)
                
                # Create benchmark visualization
                fig_bench = go.Figure()
                
                # Add regional performance bars
                fig_bench.add_bar(
                    x=regional_performance.index,
                    y=regional_performance['composite_score'],
                    name="Regional Performance",
                    marker_color='lightblue'
                )
                
                # Add benchmark lines based on selected type
                if benchmark_type == "Against National Average":
                    fig_bench.add_hline(y=national_avg, line_dash="dash", 
                                       line_color="red", 
                                       annotation_text="National Average")
                
                elif benchmark_type == "Top Performers Only":
                    fig_bench.add_hline(y=top_quartile, line_dash="dash", 
                                       line_color="green", 
                                       annotation_text="Top Quartile")
                
                elif benchmark_type == "Performance Quartiles":
                    fig_bench.add_hline(y=regional_performance['composite_score'].quantile(0.25), 
                                       line_dash="dot", line_color="orange", 
                                       annotation_text="Q1")
                    fig_bench.add_hline(y=regional_performance['composite_score'].quantile(0.5), 
                                       line_dash="dash", line_color="blue", 
                                       annotation_text="Median")
                    fig_bench.add_hline(y=top_quartile, line_dash="dot", 
                                       line_color="green", annotation_text="Q3")
                
                # Add performance threshold line
                fig_bench.add_hline(y=performance_threshold_value, line_dash="dashdot", 
                                   line_color="purple", 
                                   annotation_text=f"Performance Threshold ({performance_threshold}%)")
                
                fig_bench.update_layout(
                    title=f"Regional Performance vs {benchmark_type}",
                    xaxis_title="Regions",
                    yaxis_title="Composite Performance Score",
                    height=500
                )
                
                st.plotly_chart(fig_bench, use_container_width=True)
                
                # Benchmark analysis summary
                above_threshold = regional_performance[regional_performance['composite_score'] >= performance_threshold_value]
                below_threshold = regional_performance[regional_performance['composite_score'] < performance_threshold_value]
                
                col_bench1, col_bench2 = st.columns(2)
                with col_bench1:
                    st.markdown(f"""
                    <div class="performance-box">
                    <strong>✅ Above Threshold:</strong><br>
                    {len(above_threshold)} regions ({len(above_threshold)/len(regional_performance)*100:.1f}%)<br>
                    <strong>Average Score:</strong> {above_threshold['composite_score'].mean():.2f if len(above_threshold) > 0 else 'N/A'}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_bench2:
                    st.markdown(f"""
                    <div class="insight-box">
                    <strong>⚠️ Below Threshold:</strong><br>
                    {len(below_threshold)} regions ({len(below_threshold)/len(regional_performance)*100:.1f}%)<br>
                    <strong>Average Score:</strong> {below_threshold['composite_score'].mean():.2f if len(below_threshold) > 0 else 'N/A'}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed benchmark comparison table
                st.markdown("##### 📋 Detailed Benchmark Analysis")
                benchmark_df = regional_performance[['composite_score']].copy()
                benchmark_df['vs_national_avg'] = (benchmark_df['composite_score'] - national_avg).round(2)
                benchmark_df['vs_top_quartile'] = (benchmark_df['composite_score'] - top_quartile).round(2)
                benchmark_df['meets_threshold'] = benchmark_df['composite_score'] >= performance_threshold_value
                
                st.dataframe(benchmark_df.rename(columns={
                    'composite_score': 'Performance Score',
                    'vs_national_avg': 'vs National Avg',
                    'vs_top_quartile': 'vs Top Quartile',
                    'meets_threshold': 'Meets Threshold'
                }), use_container_width=True)
    
    else:
        st.warning("⚠️ No regional performance data available with current filters and metrics selection.")

else:
    st.info("🔍 Select regions and metrics in the sidebar to see comprehensive regional analysis.")

# Distribution Analysis (existing section)
st.markdown('<div class="sub-header">📈 Visualization 3: Distribution Analysis</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This box plot shows the distribution and variability of the selected metric across different categories.
It helps identify outliers, quartiles, and overall spread of the data.
</div>
""", unsafe_allow_html=True)

# Choose distribution column
dist_candidates = []
if col_total_hotels and col_total_hotels in numeric_cols:
    dist_candidates.append(col_total_hotels)
if metric not in dist_candidates:
    dist_candidates.append(metric)
for n in numeric_cols:
    if n not in dist_candidates:
        dist_candidates.append(n)

dist_col = st.selectbox("🎯 Select column for distribution analysis:", dist_candidates, 
                       help="Choose which numeric variable to analyze the distribution of")

if col_initiative and dist_col:
    # Prepare data for box plot
    df_box = df_filtered[[col_initiative, dist_col]].copy()
    df_box = df_box.dropna()
    
    # Ensure numeric
    try:
        df_box[dist_col] = pd.to_numeric(df_box[dist_col], errors='coerce')
        df_box = df_box.dropna(subset=[dist_col])
    except:
        pass
    
    if not df_box.empty:
        # Create enhanced box plot
        fig2 = px.box(
            df_box, 
            x=col_initiative, 
            y=dist_col,
            title=f"Distribution of {dist_col} by {col_initiative}",
            labels={col_initiative: col_initiative, dist_col: dist_col},
            color=col_initiative,
            points="outliers"  # Show outliers
        )
        
        # Add mean markers
        mean_values = df_box.groupby(col_initiative)[dist_col].mean().reset_index()
        fig2.add_scatter(
            x=mean_values[col_initiative],
            y=mean_values[dist_col],
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            name='Mean',
            showlegend=True
        )
        
        fig2.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Distribution insights
        stats = df_box.groupby(col_initiative)[dist_col].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(stats, use_container_width=True)
        
    else:
        st.warning("⚠️ No valid numeric data available for the selected distribution column and filters.")

# Additional Analysis Section
st.markdown('<div class="sub-header">🔍 Additional Interactive Analysis</div>', 
            unsafe_allow_html=True)

analysis_type = st.selectbox(
    "Choose additional analysis:",
    ["Summary Statistics", "Correlation Analysis", "Top Performers", "Regional Insights"]
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
    if col_governorate and metric and len(df_filtered) > 0:
        top_regions = df_filtered.groupby(col_governorate)[metric].mean().nlargest(5)
        col_a, col_b = st.columns(2)
        with col_a:
            st.bar_chart(top_regions)
        with col_b:
            st.write("**Top 5 Regions:**")
            for region, value in top_regions.items():
                st.write(f"• {region}: {value:.2f}")

elif analysis_type == "Regional Insights":
    st.markdown("### 🗺️ Advanced Regional Performance Insights")
    
    if col_governorate and len(df_filtered) > 0:
        geo_insights = df_filtered.groupby(col_governorate)[metric].agg(['mean', 'count', 'std']).round(2)
        geo_insights.columns = ['Average', 'Data Points', 'Variability']
        geo_insights = geo_insights.sort_values('Average', ascending=False)
        
        # Calculate advanced regional statistics
        top_region = geo_insights.index[0]
        top_value = geo_insights.iloc[0]['Average']
        bottom_region = geo_insights.index[-1]
        bottom_value = geo_insights.iloc[-1]['Average']
        
        # Performance consistency analysis
        most_consistent = geo_insights.loc[geo_insights['Variability'].idxmin()]
        least_consistent = geo_insights.loc[geo_insights['Variability'].idxmax()]
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            st.markdown(f"""
            <div class="performance-box">
            <strong>🏆 Performance Leaders:</strong><br>
            <strong>Best:</strong> {top_region} ({top_value:.2f})<br>
            <strong>Most Consistent:</strong> {most_consistent.name} (σ={most_consistent['Variability']:.2f})<br>
            <strong>🔻 Needs Attention:</strong> {bottom_region} ({bottom_value:.2f})
            </div>
            """, unsafe_allow_html=True)
        
        with col_insight2:
            performance_gap = top_value - bottom_value
            avg_variability = geo_insights['Variability'].mean()
            total_data_points = geo_insights['Data Points'].sum()
            
            st.markdown(f"""
            <div class="geographic-box">
            <strong>📊 Performance Analytics:</strong><br>
            <strong>Performance Gap:</strong> {performance_gap:.2f}<br>
            <strong>Avg Variability:</strong> {avg_variability:.2f}<br>
            <strong>Total Data Points:</strong> {total_data_points:,}<br>
            <strong>Regions Analyzed:</strong> {len(geo_insights)}
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced regional performance table with rankings
        geo_insights_ranked = geo_insights.copy()
        geo_insights_ranked['Performance_Rank'] = range(1, len(geo_insights_ranked) + 1)
        geo_insights_ranked['Consistency_Rank'] = geo_insights_ranked['Variability'].rank()
        
        st.markdown("#### 📋 Complete Regional Performance Matrix")
        st.dataframe(geo_insights_ranked, use_container_width=True)
        
        # Regional performance visualization
        fig_regional_compare = px.bar(
            geo_insights.reset_index(),
            x=col_governorate,
            y='Average',
            color='Variability',
            size='Data Points',
            title=f'Regional Performance Comparison: {metric}',
            hover_data=['Variability', 'Data Points'],
            color_continuous_scale='viridis'
        )
        fig_regional_compare.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig_regional_compare, use_container_width=True)

# Interactive Feature Highlights
st.markdown("---")
st.markdown("""
### 🚀 Interactive Features Demonstration

**🎛️ Feature 1: Performance Benchmarking Controls**
- **Location:** Regional Analysis Controls → Benchmark Analysis dropdown
- **Impact:** Changes benchmark lines in Visualization 2, Tab 4 (Benchmarks)
- **Options:** National Average, Top Performers, Regional Clusters, Performance Quartiles
- **Insight Generation:** Switch between "Against National Average" and "Performance Quartiles" to see how regions perform against different standards

**📊 Feature 2: Multi-Metric Regional Selection**
- **Location:** Regional Analysis Controls → Select Metrics for Regional Analysis
- **Impact:** Updates all tabs in Visualization 2 (Performance Matrix, Scatter Analysis, Rankings)
- **Dynamic Effect:** Heatmap colors change, scatter plot axes update, composite scores recalculate
- **Real-time Analysis:** Select different metric combinations to discover regional patterns

### 💡 How to Generate Insights:

1. **Performance Benchmarking Workflow:**
   - Select "Top Performers Only" to identify benchmark regions
   - Switch to "Performance Quartiles" to see distribution spread
   - Adjust Performance Threshold slider to set custom standards
   - Observe how regions move above/below threshold lines

2. **Multi-Metric Analysis Workflow:**
   - Start with 2-3 core tourism metrics
   - Use Scatter Analysis tab to find performance correlations
   - Check Performance Matrix for comprehensive regional comparison
   - Switch metrics to validate patterns across different indicators

3. **Regional Grouping Intelligence:**
   - Toggle between "Individual Regions" and "Performance Tiers"
   - Use clustering to identify similar-performing regional groups
   - Combine with geographic filters for targeted analysis

### 🎯 Key Design Decisions:

- **Interactive Filter Persistence**: Session state maintains selections across interactions
- **Multi-Level Geographic Analysis**: Governorate → Area hierarchy with automatic filtering
- **Real-Time Benchmarking**: Dynamic threshold lines that respond to slider changes
- **Composite Scoring**: Automatic calculation of performance indices from selected metrics
- **Clustering Integration**: ML-powered regional grouping for pattern discovery
- **Progressive Disclosure**: Tabbed interface reveals complexity gradually
""")

# Export functionality
st.markdown("### 📥 Export and Sharing")

col_export1, col_export2 = st.columns(2)

with col_export1:
    if st.button("📊 Export Filtered Data"):
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

with col_export2:
    if col_governorate and regional_metrics and not df_filtered.empty:
        regional_analysis = create_regional_performance_matrix(df_filtered, regional_metrics, col_governorate)
        if not regional_analysis.empty:
            if st.button("📈 Export Regional Analysis"):
                regional_csv = regional_analysis.to_csv()
                st.download_button(
                    label="Download Regional Performance Analysis",
                    data=regional_csv,
                    file_name=f"regional_performance_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# Footer with usage instructions
st.markdown("---")
st.markdown("""
<div class="context-box">
<strong>🎥 Video Demonstration Points:</strong><br>
1. Show Performance Benchmarking: Switch between benchmark types and highlight threshold changes<br>
2. Demonstrate Multi-Metric Selection: Change metrics and show real-time updates across all tabs<br>
3. Highlight Regional Insights: Use scatter analysis to identify high-performing vs improvement-needed regions<br>
4. Explain Design Decisions: Multi-level filtering, progressive disclosure, and ML-powered clustering
</div>
""", unsafe_allow_html=True)
