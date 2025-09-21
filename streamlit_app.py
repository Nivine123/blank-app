import plotly.express as px
import pandas as pd

# Filter the data to include only rows where initiatives exist
initiatives_exist_df = df[df['Existence of initiatives and projects in the past five years to improve the tourism sector - exists'] == 1]

# Count the number of initiatives per region
initiative_counts_by_region = initiatives_exist_df['refArea'].value_counts().reset_index()
initiative_counts_by_region.columns = ['refArea', 'Number of Initiatives']

# Now, we will also retrieve the tourism index for each region (if available)
# Group by region and calculate the average tourism index for regions with initiatives
tourism_index_by_region = initiatives_exist_df.groupby('refArea')[col_tourism_index].mean().reset_index()
tourism_index_by_region.columns = ['refArea', 'Average Tourism Index']

# Merge the two dataframes to include both the count of initiatives and the tourism index
merged_df = pd.merge(initiative_counts_by_region, tourism_index_by_region, on='refArea')

# Create a bar chart to show the number of initiatives and average tourism index by region
fig = px.bar(merged_df,
             x='refArea',
             y='Number of Initiatives',
             title='Number of Tourism Initiatives and Average Tourism Index by Region',
             labels={'refArea': 'Region', 'Number of Initiatives': 'Number of Initiatives'},
             color='Average Tourism Index',
             color_continuous_scale='Viridis',
             hover_data=['Average Tourism Index'])

fig.update_layout(barmode='group', xaxis={'tickangle': 45})
fig.show()
