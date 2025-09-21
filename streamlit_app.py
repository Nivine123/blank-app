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
