# Enhanced Region/Area Filtering Section
# Add this to your existing streamlit app

# Multi-level geographic filtering
st.sidebar.markdown("## 🗺️ Geographic Filters")

# Primary region filter (your existing governorate filter)
if col_governorate:
    uniq_gov = sorted(df[col_governorate].dropna().unique().tolist())
    
    # Option 1: Select All/None buttons
    col_all, col_none = st.sidebar.columns(2)
    with col_all:
        select_all_regions = st.button("✅ Select All Regions", key="select_all")
    with col_none:
        deselect_all_regions = st.button("❌ Deselect All", key="deselect_all")
    
    # Handle select all/none
    if select_all_regions:
        st.session_state.governorate_choice = uniq_gov
    elif deselect_all_regions:
        st.session_state.governorate_choice = []
    
    # Main region selector with enhanced features
    governorate_choice = st.sidebar.multiselect(
        f"🏛️ Select {col_governorate}s",
        options=uniq_gov,
        default=st.session_state.get('governorate_choice', uniq_gov),
        help="Filter analysis by specific governorates/regions",
        key="gov_filter"
    )
    
    # Update session state
    st.session_state.governorate_choice = governorate_choice
    
    # Show selection summary
    if len(governorate_choice) < len(uniq_gov):
        st.sidebar.info(f"📊 Selected: {len(governorate_choice)} of {len(uniq_gov)} regions")

# Secondary area filter (if you have sub-regions, cities, etc.)
col_area = find_col(df, ["Area", "City", "Municipality", "District", "Caza", "area", "city"])
area_choice = None

if col_area:
    st.sidebar.markdown("### 🏘️ Sub-Area Filter (Optional)")
    
    # Filter areas based on selected governorates
    if governorate_choice:
        available_areas = df[df[col_governorate].isin(governorate_choice)][col_area].dropna().unique()
    else:
        available_areas = df[col_area].dropna().unique()
    
    available_areas = sorted(available_areas.tolist())
    
    if available_areas:
        area_choice = st.sidebar.multiselect(
            f"🏘️ Select {col_area}s",
            options=available_areas,
            default=available_areas,
            help="Further filter by specific areas within selected regions"
        )
        
        if len(area_choice) < len(available_areas):
            st.sidebar.info(f"📊 Selected: {len(area_choice)} of {len(available_areas)} areas")

# Geographic comparison mode
st.sidebar.markdown("### 🔍 Analysis Mode")
geo_analysis_mode = st.sidebar.selectbox(
    "Geographic Analysis Focus:",
    ["All Selected Regions", "Compare Regions", "Regional Ranking", "Geographic Distribution"],
    help="Choose how to analyze the geographic data"
)

# Apply enhanced geographic filtering
df_filtered = df.copy()

# Apply governorate filter
if governorate_choice is not None and len(governorate_choice) > 0:
    df_filtered = df_filtered[df_filtered[col_governorate].isin(governorate_choice)]

# Apply area filter
if area_choice is not None and len(area_choice) > 0 and col_area:
    df_filtered = df_filtered[df_filtered[col_area].isin(area_choice)]

# Enhanced geographic insights
st.markdown("### 🗺️ Geographic Analysis Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    regions_analyzed = df_filtered[col_governorate].nunique() if col_governorate else 0
    st.metric("🏛️ Regions in Analysis", regions_analyzed)

with col2:
    if col_area and area_choice:
        areas_analyzed = df_filtered[col_area].nunique()
        st.metric("🏘️ Sub-areas Analyzed", areas_analyzed)
    else:
        total_entries = len(df_filtered)
        st.metric("📊 Total Data Points", f"{total_entries:,}")

with col3:
    if col_governorate and len(governorate_choice) > 0:
        coverage_pct = (len(governorate_choice) / len(uniq_gov)) * 100
        st.metric("📈 Regional Coverage", f"{coverage_pct:.1f}%")

with col4:
    if metric in df_filtered.columns:
        regional_avg = df_filtered[metric].mean()
        st.metric(f"🎯 Avg {metric}", f"{regional_avg:.2f}" if not pd.isna(regional_avg) else "N/A")

# Geographic-specific visualizations based on analysis mode
if geo_analysis_mode == "Compare Regions" and col_governorate:
    st.markdown("#### 🏛️ Regional Comparison")
    
    # Create side-by-side comparison
    regional_stats = df_filtered.groupby(col_governorate)[metric].agg(['mean', 'count', 'std']).round(2)
    regional_stats.columns = ['Average', 'Count', 'Std Dev']
    
    # Horizontal bar chart for better region name visibility
    fig_regional = px.bar(
        regional_stats.reset_index(), 
        x='Average', 
        y=col_governorate,
        orientation='h',
        title=f"Regional Comparison: Average {metric}",
        color='Average',
        color_continuous_scale="viridis"
    )
    fig_regional.update_layout(height=max(400, len(governorate_choice) * 50))
    st.plotly_chart(fig_regional, use_container_width=True)
    
    # Show detailed comparison table
    st.dataframe(regional_stats, use_container_width=True)

elif geo_analysis_mode == "Regional Ranking":
    st.markdown("#### 🏆 Regional Performance Ranking")
    
    if col_governorate and metric:
        ranking_df = df_filtered.groupby(col_governorate).agg({
            metric: ['mean', 'sum', 'count']
        }).round(2)
        
        ranking_df.columns = ['Average', 'Total', 'Count']
        ranking_df = ranking_df.sort_values('Average', ascending=False).reset_index()
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        # Reorder columns
        ranking_df = ranking_df[['Rank', col_governorate, 'Average', 'Total', 'Count']]
        
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)

elif geo_analysis_mode == "Geographic Distribution" and col_governorate:
    st.markdown("#### 📍 Geographic Distribution Analysis")
    
    # Create a more detailed geographic breakdown
    if col_area and area_choice:
        # Two-level geographic analysis
        geo_df = df_filtered.groupby([col_governorate, col_area])[metric].mean().reset_index()
        
        fig_geo = px.sunburst(
            geo_df,
            path=[col_governorate, col_area],
            values=metric,
            title=f"Hierarchical Geographic Distribution of {metric}"
        )
        st.plotly_chart(fig_geo, use_container_width=True)
    else:
        # Single-level geographic pie chart
        geo_totals = df_filtered.groupby(col_governorate)[metric].sum()
        
        fig_pie = px.pie(
            values=geo_totals.values,
            names=geo_totals.index,
            title=f"Geographic Distribution of Total {metric}"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Geographic insights box
if len(df_filtered) > 0 and col_governorate:
    top_region = df_filtered.groupby(col_governorate)[metric].mean().idxmax()
    top_value = df_filtered.groupby(col_governorate)[metric].mean().max()
    
    bottom_region = df_filtered.groupby(col_governorate)[metric].mean().idxmin()
    bottom_value = df_filtered.groupby(col_governorate)[metric].mean().min()
    
    st.markdown(f"""
    <div class="insight-box">
    <strong>🗺️ Geographic Insights:</strong><br>
    • <strong>Top Performing Region:</strong> {top_region} (Avg: {top_value:.2f})<br>
    • <strong>Lowest Performing Region:</strong> {bottom_region} (Avg: {bottom_value:.2f})<br>
    • <strong>Performance Gap:</strong> {top_value - bottom_value:.2f}<br>
    • <strong>Regional Variability:</strong> {df_filtered.groupby(col_governorate)[metric].mean().std():.2f}
    </div>
    """, unsafe_allow_html=True)
