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

# NEW: Regional Analysis Interactive Controls
st.sidebar.markdown("## 🗺️ Regional Analysis Controls")

# Interactive Feature 1: Visualization Type Selector
viz_type = st.sidebar.selectbox(
    "🎯 Regional Visualization Type:",
    ["Bar Chart Comparison", "Scatter Plot Analysis", "Pie Chart Distribution", "Box Plot by Region"],
    help="Choose different ways to visualize regional data"
)

# Interactive Feature 2: Metric Comparison Mode
comparison_mode = st.sidebar.selectbox(
    "📊 Regional Comparison Mode:",
    ["Single Metric Analysis", "Two Metric Comparison", "Multi-Metric Overview"],
    help="Select how many metrics to compare across regions"
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

# Visualization 1: Aggregated Analysis (ORIGINAL)
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

# NEW: Visualization 2: Interactive Regional Analysis
st.markdown('<div class="sub-header">🗺️ Visualization 2: Interactive Regional Performance Analysis</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This interactive regional analysis provides insights into tourism performance 
across different geographic regions using multiple visualization types and comparison modes.
The two interactive features allow you to switch between different chart types and comparison methods.
</div>
""", unsafe_allow_html=True)

if col_governorate and governorate_choice and len(df_filtered) > 0:
    
    # Prepare regional data based on comparison mode
    if comparison_mode == "Single Metric Analysis":
        # Simple regional aggregation
        regional_data = df_filtered.groupby(col_governorate)[metric].agg(['mean', 'count']).reset_index()
        regional_data.columns = ['Region', 'Average_Value', 'Data_Points']
        
    elif comparison_mode == "Two Metric Comparison" and len(preferred_metrics) >= 2:
        # Two metric comparison
        metric_2 = st.selectbox("📈 Select Second Metric for Comparison:", 
                               [m for m in preferred_metrics if m != metric], 
                               key="second_metric")
        
        regional_data = df_filtered.groupby(col_governorate).agg({
            metric: 'mean',
            metric_2: 'mean'
        }).reset_index()
        regional_data.columns = ['Region', 'Metric_1', 'Metric_2']
        
    else:  # Multi-Metric Overview
        # Use top 3 metrics
        top_metrics = preferred_metrics[:3]
        agg_dict = {m: 'mean' for m in top_metrics}
        regional_data = df_filtered.groupby(col_governorate).agg(agg_dict).reset_index()
        
    # Create visualization based on selected type
    if viz_type == "Bar Chart Comparison":
        if comparison_mode == "Single Metric Analysis":
            fig2 = px.bar(
                regional_data.sort_values('Average_Value', ascending=False),
                x='Region',
                y='Average_Value',
                title=f'Regional Comparison: {metric} by {col_governorate}',
                color='Average_Value',
                color_continuous_scale='viridis',
                text='Average_Value'
            )
            fig2.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            
        elif comparison_mode == "Two Metric Comparison" and len(preferred_metrics) >= 2:
            fig2 = px.bar(
                regional_data,
                x='Region',
                y=['Metric_1', 'Metric_2'],
                title=f'Regional Comparison: {metric} vs {metric_2}',
                barmode='group'
            )
        
        else:  # Multi-metric
            fig2 = px.bar(
                regional_data,
                x='Region',
                y=top_metrics,
                title=f'Multi-Metric Regional Comparison',
                barmode='group'
            )
    
    elif viz_type == "Scatter Plot Analysis" and comparison_mode == "Two Metric Comparison" and len(preferred_metrics) >= 2:
        fig2 = px.scatter(
            regional_data,
            x='Metric_1',
            y='Metric_2',
            hover_data=['Region'],
            title=f'Regional Scatter: {metric} vs {metric_2}',
            color='Region',
            size_max=15
        )
        
        # Add region labels
        for i, row in regional_data.iterrows():
            fig2.add_annotation(
                x=row['Metric_1'], y=row['Metric_2'],
                text=row['Region'], 
                showarrow=False,
                font=dict(size=10)
            )
    
    elif viz_type == "Pie Chart Distribution":
        if comparison_mode == "Single Metric Analysis":
            fig2 = px.pie(
                regional_data,
                values='Average_Value',
                names='Region',
                title=f'Regional Distribution of {metric}'
            )
    
    elif viz_type == "Box Plot by Region":
        fig2 = px.box(
            df_filtered,
            x=col_governorate,
            y=metric,
            title=f'Distribution of {metric} by Region',
            color=col_governorate
        )
        fig2.update_layout(xaxis_tickangle=45)
    
    # Default fallback - simple bar chart
    if 'fig2' not in locals():
        regional_summary = df_filtered.groupby(col_governorate)[metric].mean().reset_index()
        fig2 = px.bar(
            regional_summary.sort_values(metric, ascending=False),
            x=col_governorate,
            y=metric,
            title=f'Regional Average: {metric}',
            color=metric,
            color_continuous_scale='plasma'
        )
    
    # Display the chart
    fig2.update_layout(height=500, xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Regional insights
    if comparison_mode == "Single Metric Analysis":
        best_region = regional_data.loc[regional_data['Average_Value'].idxmax(), 'Region']
        best_value = regional_data['Average_Value'].max()
        worst_region = regional_data.loc[regional_data['Average_Value'].idxmin(), 'Region']
        worst_value = regional_data['Average_Value'].min()
        
        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            st.markdown(f"""
            <div class="geographic-box">
            <strong>🏆 Best Performer:</strong><br>
            {best_region}<br>
            <strong>Score:</strong> {best_value:.2f}
            </div>
            """, unsafe_allow_html=True)
        
        with col_i2:
            st.markdown(f"""
            <div class="insight-box">
            <strong>📊 Performance Gap:</strong><br>
            {best_value - worst_value:.2f}<br>
            <strong>Regions:</strong> {len(regional_data)}
            </div>
            """, unsafe_allow_html=True)
        
        with col_i3:
            st.markdown(f"""
            <div class="context-box">
            <strong>🎯 Needs Focus:</strong><br>
            {worst_region}<br>
            <strong>Score:</strong> {worst_value:.2f}
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("🔍 Select regions in the sidebar to see regional analysis.")

# Rest of your original code (Distribution Analysis, Additional Analysis, etc.)
# [Keep all your existing sections from the original code here]

# Distribution Analysis
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
        fig3 = px.box(
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
        fig3.add_scatter(
            x=mean_values[col_initiative],
            y=mean_values[dist_col],
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            name='Mean',
            showlegend=True
        )
        
        fig3.update_layout(height=500, showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Distribution insights
        stats = df_box.groupby(col_initiative)[dist_col].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(stats, use_container_width=True)
        
    else:
        st.warning("⚠️ No valid numeric data available for the selected distribution column and filters.")

# Interactive Features Summary
st.markdown("---")
st.markdown("""
### 🚀 Interactive Features Summary

**🎛️ Feature 1: Visualization Type Selector**
- **Location**: Sidebar → Regional Analysis Controls → "Regional Visualization Type"
- **Impact**: Changes the chart type in Visualization 2 (Bar Chart, Scatter Plot, Pie Chart, Box Plot)
- **Real-time Effect**: Instantly updates the regional visualization with different chart styles

**📊 Feature 2: Comparison Mode Selector**  
- **Location**: Sidebar → Regional Analysis Controls → "Regional Comparison Mode"
- **Impact**: Changes data structure and analysis depth (Single Metric, Two Metric, Multi-Metric)
- **Dynamic Changes**: Updates data aggregation, chart content, and available visualization options

### 💡 How to Use for Insights:

1. **Chart Type Exploration**: Switch between Bar Chart and Box Plot to see regional performance vs distribution
2. **Comparison Depth**: Use "Two Metric Comparison" with Scatter Plot to find regional correlations  
3. **Distribution Analysis**: Box Plot shows which regions have consistent vs variable performance
4. **Regional Patterns**: Pie Chart reveals regional contribution proportions

### 🎯 For Your Video (1 minute):
- **0-15s**: Show Visualization Type switching (Bar → Scatter → Box Plot)
- **15-30s**: Demonstrate Comparison Mode changes (Single → Two Metric)  
- **30-45s**: Highlight regional insights generated
- **45-60s**: Explain design decisions (interactive flexibility, multiple perspectives)
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
        file_name=f"
