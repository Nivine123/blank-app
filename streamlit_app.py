import streamlit as st
import pandas as pd
import plotly.express as px

# Assuming df is already loaded as per your original code

# Inspect the column names in the dataset
st.write("Columns in the DataFrame:")
st.write(df.columns.tolist())

# Update these column names according to what you find in the previous step
column_initiatives = 'Existence of initiatives and projects in the past five years to improve the tourism sector - exists'
column_tourism_index = 'Tourism Index'
column_ref_area = 'refArea'

# Check if the column names are correct
if column_initiatives not in df.columns:
    st.error(f"Column '{column_initiatives}' not found in the dataset. Please check the column name.")
else:
    # Filter the data to include only rows where initiatives exist
    initiatives_exist_df = df[df[column_initiatives] == 1]

    # Count the number of initiatives per region
    initiative_counts_by_region = initiatives_exist_df[column_ref_area].value_counts().reset_index()
    initiative_counts_by_region.columns = ['Region', 'Number of Initiatives']

    # Group by region and calculate the average tourism index for regions with initiatives
    tourism_index_by_region = initiatives_exist_df.groupby(column_ref_area)[column_tourism_index].mean().reset_index()
    tourism_index_by_region.columns = ['Region', 'Average Tourism Index']

    # Merge the two dataframes to include both the count of initiatives and the tourism index
    merged_df = pd.merge(initiative_counts_by_region, tourism_index_by_region, on='Region')

    # Create a bar chart to show the number of initiatives and average tourism index by region
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

    # Update the layout of the chart for better readability
    fig.update_layout(
        barmode='group',
        xaxis={'tickangle': 45},  # Rotate region names for better readability
        showlegend=False
    )

    # Show the plot
    st.plotly_chart(fig)
