# Ensure that 'col_area' is properly initialized based on the dataset
col_area = find_col(df, ["Area", "City", "Municipality", "District", "Caza", "area", "city"])

# If 'col_area' is found, proceed with the filters
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
    else:
        st.warning("⚠️ No areas found in the dataset for the selected governorates.")
else:
    st.warning("⚠️ 'Area' column is missing or not found in the dataset.")
