# Ensure this block starts with a proper condition
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

# Check if the geo_analysis_mode is "Regional Ranking"
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
        ranking_df = ranking_df[['Rank', col_governorate, 'Average', 'Total', 'Count', 'Performance']]
        
        # Display the ranking data
        st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    else:
        st.info("🔍 Select multiple regions to see ranking comparison")
