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
region_col = col_governorate or col_ref_area

if region_col:
    st.sidebar.markdown("### 🏛️ Primary Region Filter")
    
    uniq_regions = sorted(df[region_col].dropna().unique().tolist())
    
    # Quick selection buttons
    col_all, col_none, col_reset = st.sidebar.columns(3)
    with col_all:
        if st.button("✅ All", key="select_all_gov", help="Select all regions"):
            st.session_state.governorate_choice = uniq_regions
            st.rerun()
    with col_none:
        if st.button("❌ None", key="deselect_all_gov", help="Deselect all regions"):
            st.session_state.governorate_choice = []
            st.rerun()
    with col_reset:
        if st.button("🔄 Reset", key="reset_gov", help="Reset to default"):
            st.session_state.governorate_choice = uniq_regions
            st.rerun()
    
    # Main region selector
    governorate_choice = st.sidebar.multiselect(
        f"🏛️ Select Regions",
        options=uniq_regions,
        default=st.session_state.governorate_choice,
        help="Filter analysis by specific regions",
        key="gov_multiselect"
    )
    
    # Update session state
    st.session_state.governorate_choice = governorate_choice
    
    # Show selection summary
    if len(governorate_choice) != len(uniq_regions):
        coverage_pct = (len(governorate_choice) / len(uniq_regions)) * 100
        st.sidebar.info(f"📊 Selected: {len(governorate_choice)}/{len(uniq_regions)} regions ({coverage_pct:.1f}%)")

# NEW: Interactive controls for regional initiatives visualization
st.sidebar.markdown("## 🎯 Initiatives Visualization Controls")

# Interactive Feature 1: Chart Type Selection for initiatives visualization
initiatives_chart_type = st.sidebar.selectbox(
    "📊 Regional Initiatives Chart Type:",
    ["Bar Chart", "Horizontal Bar Chart", "Pie Chart", "Donut Chart"],
    help="Choose how to display regional initiatives data"
)

# Interactive Feature 2: Tourism Index Integration
show_tourism_index = st.sidebar.checkbox(
    "📈 Include Tourism Index in Regional Initiatives Analysis",
    value=True,
    help="Add tourism index as color coding in regional initiatives visualization"
)

# Secondary area filter (if available)
area_choice = None
if col_area:
    st.sidebar.markdown("### 🏘️ Sub-Area Filter")
    
    # Filter areas based on selected regions
    if governorate_choice and len(governorate_choice) > 0:
        available_areas = df[df[region_col].isin(governorate_choice)][col_area].dropna().unique()
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

# Apply region filter
if governorate_choice is not None and len(governorate_choice) > 0:
    df_filtered = df_filtered[df_filtered[region_col].isin(governorate_choice)]
    filter_steps.append(f"Regions: {len(governorate_choice)} selected")

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
    if region_col and governorate_choice:
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
    if col_initiative and selected_initiatives:
        # Count records with initiatives = 1
        if 1 in df_filtered[col_initiative].values:
            initiative_count = len(df_filtered[df_filtered[col_initiative] == 1])
        else:
            initiative_count = len(selected_initiatives)
        st.metric("🏗️ Active Initiatives", initiative_count)

# ENHANCED GEOGRAPHIC ANALYSIS SECTION
if region_col and governorate_choice and geo_analysis_mode != "Standard Analysis":
    st.markdown('<div class="sub-header">🗺️ Geographic Analysis Results</div>', unsafe_allow_html=True)
    
    if geo_analysis_mode == "Compare Regions" and len(governorate_choice) > 1:
        st.markdown("""
        <div class="geographic-box">
        <strong>🔍 Regional Comparison Mode:</strong> Comparing performance across selected regions
        </div>
        """, unsafe_allow_html=True)
        
        # Regional comparison analysis
        regional_stats = df_filtered.groupby(region_col)[metric].agg(['mean', 'count', 'std']).round(2)
        regional_stats.columns = ['Average', 'Count', 'Std Dev']
        
        # Create horizontal bar chart for better region name visibility
        fig_regional = px.bar(
            regional_stats.reset_index(), 
            x='Average', 
            y=region_col,
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
            ranking_df = df_filtered.groupby(region_col).agg({
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
            ranking_df = ranking_df[['Rank', region_col, 'Average', 'Total', 'Count', 'Performance']]
            
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        else:
            st.info("🔍 Select multiple regions to see ranking comparison")
    
    elif geo_analysis_mode == "Geographic Distribution":
        st.markdown("""
        <div class="geographic-box">
        <strong>📍 Geographic Distribution Mode:</strong> Visualizing data distribution across geography
        </div>
        """, unsafe_allow_html=True)
        
        if col_area and area_choice:
            # Two-level geographic analysis
            geo_df = df_filtered.groupby([region_col, col_area])[metric].mean().reset_index()
            
            if not geo_df.empty:
                fig_geo = px.sunburst(
                    geo_df,
                    path=[region_col, col_area],
                    values=metric,
                    title=f"Hierarchical Geographic Distribution of {metric}"
                )
                st.plotly_chart(fig_geo, use_container_width=True)
        else:
            # Single-level geographic pie chart
            geo_totals = df_filtered.groupby(region_col)[metric].sum()
            
            fig_pie = px.pie(
                values=geo_totals.values,
                names=geo_totals.index,
                title=f"Geographic Distribution of Total {metric}"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

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
    
    # Enhanced insights with geographic context
    insights_text = f"""
    <div class="insight-box">
    <strong>💡 Key Insights:</strong><br>
    • Highest {agg_func}: <strong>{agg_df.iloc[0][col_initiative]}</strong> with {agg_df.iloc[0][metric]:.2f}<br>
    • Lowest {agg_func}: <strong>{agg_df.iloc[-1][col_initiative]}</strong> with {agg_df.iloc[-1][metric]:.2f}<br>
    • Range: {agg_df[metric].max() - agg_df[metric].min():.2f} ({metric})
    """
    
    if region_col and len(governorate_choice) > 0:
        top_region = df_filtered.groupby(region_col)[metric].mean().idxmax()
        insights_text += f"<br>• Top performing region: <strong>{top_region}</strong>"
    
    insights_text += "</div>"
    st.markdown(insights_text, unsafe_allow_html=True)

else:
    st.error("❌ Cannot create visualization - no valid categorical column or empty dataset.")

# NEW: Visualization 2 - Tourism Initiatives by Region
st.markdown('<div class="sub-header">🗺️ Visualization 2: Tourism Initiatives by Region</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div class="context-box">
<strong>🎯 Purpose:</strong> This visualization shows the number of tourism initiatives across different regions.
The interactive features allow you to explore the data from multiple perspectives and understand the relationship 
between initiative activity and tourism performance across regions.
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
        
        # Sort by number of initiatives (descending)
        regional_data = regional_data.sort_values('Number of Initiatives', ascending=False)
        
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
            # Sort for better readability in horizontal layout
            regional_data_sorted = regional_data.sort_values('Number of Initiatives', ascending=True)
            
            if show_tourism_index and 'Average Tourism Index' in regional_data.columns:
                fig2 = px.bar(
                    regional_data_sorted,
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
                    regional_data_sorted,
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
        
    else:
        st.warning("⚠️ No regions with active initiatives found in the filtered data.")

# Visualization 3: Distribution Analysis (from original code)
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

# Footer with insights and instructions
st.markdown("---")
st.markdown("""
### 🚀 How to Use This Enhanced Dashboard:

1. **Enhanced Geographic Filtering**: 
   - Use quick selection buttons (All/None/Reset) for efficient region management
   - Multi-level filtering: Governorate → Area for detailed geographic analysis
   - Choose geographic analysis modes for different perspectives

2. **Interactive Features**:
   - **Feature 1**: Dynamic chart type selection for initiatives visualization (Bar, Horizontal Bar, Pie, Donut)
   - **Feature 2**: Tourism index integration toggle to add performance context to initiative analysis

3. **Data Analysis**: 
   - Switch between different tourism metrics and aggregation methods
   - Examine data distribution and identify outliers using enhanced visualizations
   - Explore correlation between initiative activity and tourism performance

### 💡 Key Design Decisions:

- **Smart Geographic Hierarchy**: Automatically detects and links governorate and area columns
- **Interactive Filter Management**: Session state preserves selections and provides quick controls
- **Multiple Analysis Perspectives**: Standard analysis plus specialized geographic modes
- **Enhanced Visual Feedback**: Coverage percentages and filter summaries
- **Comprehensive Insights**: Combines statistical analysis with geographic intelligence
- **Regional Initiative Focus**: New visualization specifically shows tourism initiatives by region with interactive chart types
""")

# Export functionality
if st.button("📥 Export Filtered Data"):
    csv = df_filtered.to_csv(index=False)
    
    # Create filename with filter info
    filter_info = []
    if governorate_choice and region_col and len(governorate_choice) < len(df[region_col].unique()):
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
